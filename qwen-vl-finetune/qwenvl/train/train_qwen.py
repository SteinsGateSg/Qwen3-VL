# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


class PromptTuningWrapper(nn.Module):
    def __init__(self, base_model, prompt_length: int):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = base_model.language_model.config.hidden_size
        embed_dtype = self.base_model.get_input_embeddings().weight.dtype
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(prompt_length, hidden_size, dtype=embed_dtype)
        )
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("Prompt tuning requires input_ids to build embeddings.")
        inputs_embeds = self.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.shape[0]
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.long, device=input_ids.device
            )
        prompt_attention = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        kwargs.pop("input_ids", None)
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, contrastive_loss_weight=0.0, contrastive_temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_temperature = contrastive_temperature

    def _unwrap_model(self, model):
        return getattr(model, "base_model", model)

    def _get_visual_features(self, model, pixel_values, image_grid_thw):
        visual = getattr(model, "visual", None)
        if visual is None:
            return None
        try:
            if image_grid_thw is not None:
                outputs = visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
            else:
                outputs = visual(pixel_values=pixel_values)
        except TypeError:
            if image_grid_thw is not None:
                outputs = visual(pixel_values, image_grid_thw)
            else:
                outputs = visual(pixel_values)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            outputs = outputs.last_hidden_state
        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        loss = outputs.loss
        if self.contrastive_loss_weight <= 0:
            return (loss, outputs) if return_outputs else loss

        base_model = self._unwrap_model(model)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        labels = inputs.get("labels")
        if pixel_values is None or labels is None:
            return (loss, outputs) if return_outputs else loss

        visual_feats = self._get_visual_features(base_model, pixel_values, image_grid_thw)
        if visual_feats is None:
            return (loss, outputs) if return_outputs else loss

        image_embeds = visual_feats.mean(dim=1)
        hidden_states = outputs.hidden_states[-1]
        text_mask = labels.ne(-100)
        if text_mask.any():
            mask = text_mask.unsqueeze(-1)
            text_embeds = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            text_embeds = hidden_states.mean(dim=1)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = image_embeds @ text_embeds.t()
        logits = logits / self.contrastive_temperature
        batch_size = logits.size(0)
        targets = torch.arange(batch_size, device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        contrastive_loss = (loss_i2t + loss_t2i) * 0.5
        loss = loss + self.contrastive_loss_weight * contrastive_loss
        return (loss, outputs) if return_outputs else loss


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable and model_args.tune_prompt:
        raise ValueError("LoRA and prompt tuning cannot be enabled together.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        if model_args.tune_prompt:
            for p in model.parameters():
                p.requires_grad = False
            model = PromptTuningWrapper(model, training_args.prompt_length)
            model.prompt_embeddings.requires_grad = True
        else:
            set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            if hasattr(model, "visual"):
                model.visual.print_trainable_parameters()
            if hasattr(model, "model"):
                model.model.print_trainable_parameters()
            if hasattr(model, "prompt_embeddings"):
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"prompt trainable params: {trainable} / {total}")
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    if training_args.contrastive_loss_weight > 0:
        trainer = ContrastiveTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            contrastive_loss_weight=training_args.contrastive_loss_weight,
            contrastive_temperature=training_args.contrastive_temperature,
            **data_module,
        )
    else:
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

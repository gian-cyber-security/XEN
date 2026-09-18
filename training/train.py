from __future__ import annotations

import argparse, torch, yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/train.yaml"); a = p.parse_args()
    with open(a.config, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"device_map": "auto"}
    if torch.cuda.is_available(): kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if cfg.get("load_in_4bit", False):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=kw.get("torch_dtype", torch.float16), bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kw)
    data = load_dataset("json", data_files=cfg["train_file"], split="train")
    data = data.map(lambda x: {"text": f"User: {x['instruction']}\nAssistant: {x['response']}"})
    l = cfg["lora"]; peft = LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"], target_modules=l["target_modules"], task_type="CAUSAL_LM")
    t = cfg["training"]
    args = TrainingArguments(output_dir=cfg["output_dir"], num_train_epochs=t["num_train_epochs"], per_device_train_batch_size=t["per_device_train_batch_size"], gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"], warmup_ratio=t["warmup_ratio"], logging_steps=t["logging_steps"], save_steps=t["save_steps"], save_total_limit=t["save_total_limit"], gradient_checkpointing=t["gradient_checkpointing"], bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(), report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, peft_config=peft, processing_class=tok, dataset_text_field="text", max_seq_length=cfg["max_seq_length"])
    trainer.train(); trainer.save_model(cfg["output_dir"]); tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import os
import torch
from model.loader import load_model


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return prompt


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
    text = build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, pad_token_id=tokenizer.pad_token_id)
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("XEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()
    bundle = load_model(args.model, load_in_4bit=args.load_in_4bit)
    print(generate(bundle.model, bundle.tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p))


if __name__ == "__main__":
    main()

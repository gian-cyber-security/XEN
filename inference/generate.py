from __future__ import annotations
import argparse
from pathlib import Path
import torch
from model.loader import load_model

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_t.txt")

def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    system_prompt = load_system_prompt()
    full_prompt = f"System: {system_prompt}\nUser: {prompt.strip()}\nAssistant:"
    ids = tokenizer.encode(full_prompt, max_length=model.cfg.max_seq_len)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        context = x[:, -model.cfg.max_seq_len:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1) if temperature > 0 else torch.argmax(probs, dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == tokenizer.vocab["<eos>"]:
            break
    return tokenizer.decode(x[0, len(ids):].tolist())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="outputs/xen")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()
    model, tokenizer = load_model(a.model_dir)
    print(generate(model, tokenizer, a.prompt, a.max_new_tokens, a.temperature))

if __name__ == "__main__":
    main()

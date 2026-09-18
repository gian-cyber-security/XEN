from __future__ import annotations

import argparse, json
from model.loader import load_model
from inference.generate import generate


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507"); p.add_argument("--data", default="evaluation/benchmarks.jsonl"); a = p.parse_args()
    b = load_model(a.model); total = exact = 0
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line); pred = generate(b.model, b.tokenizer, item["prompt"], 256, 0, 1); total += 1
            expected = item.get("expected", "").strip()
            if expected and pred.strip() == expected: exact += 1
            print(f"PROMPT: {item['prompt']}\nPREDICTION: {pred}\nEXPECTED: {expected}\n")
    if total: print(f"Exact-match: {exact}/{total} ({exact/total:.1%})")


if __name__ == "__main__": main()

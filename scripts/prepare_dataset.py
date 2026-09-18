from __future__ import annotations

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "train.jsonl"
    count = 0
    with open(a.input, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        for n, line in enumerate(src, 1):
            if not line.strip(): continue
            item = json.loads(line)
            instruction = str(item.get("instruction", "")).strip()
            response = str(item.get("response", "")).strip()
            if not instruction or not response:
                raise ValueError(f"Line {n}: instruction and response are required")
            dst.write(json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False) + "\n")
            count += 1
    print(f"Validated {count} examples -> {target}")


if __name__ == "__main__": main()

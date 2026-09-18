from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model.image_model import XENImageModel
from model.image_conditioner import XENImageTextEncoder
from model.tokenizer import XENTokenizer

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_i.txt")

def load_system_prompt():
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

class ImageTextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, size=256):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "image" not in item or "caption" not in item:
                    raise ValueError(f"Line {line_no}: expected image and caption")
                self.rows.append((item["image"], item["caption"]))
        self.tokenizer = tokenizer
        self.size = size
        self.system_prompt = load_system_prompt()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_path, caption = self.rows[idx]
        image = Image.open(image_path).convert("RGB").resize((self.size, self.size))
        x = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        text = self.system_prompt + "\n\nUSER VISUAL REQUEST:\n" + str(caption).strip()
        ids = torch.tensor(self.tokenizer.encode(text, max_length=256), dtype=torch.long)
        return x, ids

def collate(batch):
    max_len = max(item[1].numel() for item in batch)
    images = torch.stack([item[0] for item in batch])
    ids = torch.stack([
        F.pad(item[1], (0, max_len - item[1].numel()), value=0)
        for item in batch
    ])
    return images, ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/image_train.jsonl")
    p.add_argument("--output", default="outputs/xen-gen1-i")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    tok = XENTokenizer()
    tok.fit(vocab_size=260)
    dataset = ImageTextDataset(a.data, tok, a.size)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XENImageModel().to(device)
    conditioner = XENImageTextEncoder().to(device)
    params = list(model.parameters()) + list(conditioner.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")

    global_step = 0
    model.train()
    conditioner.train()

    while global_step < a.steps:
        for images, ids in loader:
            images, ids = images.to(device), ids.to(device)
            b = images.shape[0]
            t = torch.randint(0, 1000, (b,), device=device)
            noise = torch.randn_like(images)
            alpha_bar = torch.cos(
                (t.float() / 999.0 + 0.008) / 1.008 * torch.pi / 2
            ).pow(2).clamp(1e-4, 0.9999)[:, None, None, None]
            noisy = alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise
            text_emb = conditioner(ids)
            pred = model(noisy, t, text_emb)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            global_step += 1
            if global_step % 20 == 0:
                print(f"step={global_step} loss={loss.item():.5f}")
            if global_step >= a.steps:
                break

    torch.save(
        {"model": model.state_dict(), "conditioner": conditioner.state_dict()},
        out / "model.pt",
    )
    print(f"Training complete. Output={out}")

if __name__ == "__main__":
    main()

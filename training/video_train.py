from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.tokenizer import XENTokenizer
from model.video_conditioner import XENVideoTextEncoder
from model.video_model import XENVideoModel

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_v.txt")


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def sample_video(path: str, frames: int, size: int, rng: random.Random) -> torch.Tensor:
    reader = iio.imiter(path, plugin="ffmpeg")
    all_frames = []
    for frame in reader:
        all_frames.append(frame)
    if not all_frames:
        raise ValueError(f"Video contains no frames: {path}")

    arr = np.asarray(all_frames)
    total = arr.shape[0]
    if total >= frames:
        start = rng.randint(0, total - frames)
        clip = arr[start : start + frames]
    else:
        indices = np.linspace(0, total - 1, frames).round().astype(np.int64)
        clip = arr[indices]

    # Nearest-neighbor frame sampling is deliberate and deterministic here.
    clip = np.stack(
        [np.asarray(torch.from_numpy(f).permute(2, 0, 1).float().numpy()) for f in clip]
    )
    x = torch.from_numpy(clip).permute(1, 0, 2, 3).float() / 127.5 - 1.0
    x = F.interpolate(x.unsqueeze(0), size=(frames, size, size), mode="trilinear", align_corners=False)
    return x.squeeze(0)


class VideoTextDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: XENTokenizer, frames: int, size: int, seed: int):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "video" not in item or "caption" not in item:
                    raise ValueError(f"Line {line_no}: expected video and caption")
                self.rows.append((item["video"], item["caption"]))
        if not self.rows:
            raise ValueError("Video dataset is empty.")
        self.tokenizer = tokenizer
        self.frames = frames
        self.size = size
        self.rng = random.Random(seed)
        self.system_prompt = load_system_prompt()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video_path, caption = self.rows[idx]
        clip = sample_video(video_path, self.frames, self.size, self.rng)
        text = self.system_prompt + "\n\nUSER VIDEO REQUEST:\n" + str(caption).strip()
        ids = torch.tensor(self.tokenizer.encode(text, max_length=512), dtype=torch.long)
        return clip, ids


def collate(batch):
    max_len = max(x[1].numel() for x in batch)
    clips = torch.stack([x[0] for x in batch])
    ids = torch.stack([F.pad(x[1], (0, max_len - x[1].numel()), value=0) for x in batch])
    return clips, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/video_data.jsonl")
    p.add_argument("--output", default="outputs/xen-gen1-v")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-accumulation", type=int, default=1)
    a = p.parse_args()

    if a.frames < 2:
        raise ValueError("--frames must be at least 2")
    if a.size < 32:
        raise ValueError("--size must be at least 32")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    tokenizer = XENTokenizer()
    tokenizer.fit(vocab_size=260)
    dataset = VideoTextDataset(a.data, tokenizer, a.frames, a.size, a.seed)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XENVideoModel().to(device)
    conditioner = XENVideoTextEncoder().to(device)
    params = list(model.parameters()) + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out / "tokenizer.json")

    model.train()
    conditioner.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    while global_step < a.steps:
        for clips, ids in loader:
            clips, ids = clips.to(device, non_blocking=True), ids.to(device, non_blocking=True)
            b = clips.shape[0]
            t = torch.randint(0, 1000, (b,), device=device)
            noise = torch.randn_like(clips)
            alpha_bar = torch.cos(
                (t.float() / 999.0 + 0.008) / 1.008 * torch.pi / 2
            ).pow(2).clamp(1e-4, 0.9999)
            alpha_bar = alpha_bar[:, None, None, None, None]
            noisy = alpha_bar.sqrt() * clips + (1.0 - alpha_bar).sqrt() * noise

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                text_emb = conditioner(ids)
                pred = model(noisy, t, text_emb)
                loss = F.mse_loss(pred, noise) / a.grad_accumulation

            scaler.scale(loss).backward()

            if global_step % a.grad_accumulation == a.grad_accumulation - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 20 == 0:
                print(f"step={global_step} loss={loss.item() * a.grad_accumulation:.5f}")
            if global_step >= a.steps:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "conditioner": conditioner.state_dict(),
            "frames": a.frames,
            "size": a.size,
            "prediction_type": "epsilon",
        },
        out / "model.pt",
    )
    print(f"Training complete. Output={out}")


if __name__ == "__main__":
    main()

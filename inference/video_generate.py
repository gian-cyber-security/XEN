from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from model.tokenizer import XENTokenizer
from model.video_conditioner import XENVideoTextEncoder
from model.video_model import XENVideoModel

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_v.txt")


def alpha_bar(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(((t.float() / 999.0) + 0.008) / 1.008 * math.pi / 2).pow(2).clamp(1e-4, 0.9999)


def save_video(video: torch.Tensor, path: str, fps: int) -> None:
    frames = ((video.clamp(-1, 1) + 1.0) * 127.5).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    iio.imwrite(path, frames, plugin="ffmpeg", fps=fps)


@torch.no_grad()
def generate(model_dir: str, prompt: str, output: str, frames: int, size: int, steps: int, seed: int, fps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=False)
    tokenizer = XENTokenizer.load(Path(model_dir) / "tokenizer.json")

    model = XENVideoModel().to(device)
    conditioner = XENVideoTextEncoder().to(device)
    model.load_state_dict(data["model"])
    conditioner.load_state_dict(data["conditioner"])
    model.eval()
    conditioner.eval()

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    text = system_prompt + "\n\nUSER VIDEO REQUEST:\n" + prompt.strip()
    ids = torch.tensor([tokenizer.encode(text, max_length=512)], device=device)
    cond = conditioner(ids)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn((1, 3, frames, size, size), device=device, generator=generator)

    times = torch.linspace(999, 0, steps, device=device)
    for i in range(len(times) - 1):
        t = times[i].expand(1)
        t_next = times[i + 1].expand(1)
        ab_t = alpha_bar(t)[:, None, None, None, None]
        ab_next = alpha_bar(t_next)[:, None, None, None, None]

        pred_eps = model(x, t, cond)
        x0 = (x - (1.0 - ab_t).sqrt() * pred_eps) / ab_t.sqrt().clamp_min(1e-4)
        x0 = x0.clamp(-1.5, 1.5)
        direction = (1.0 - ab_next).sqrt() * pred_eps
        x = ab_next.sqrt() * x0 + direction

    save_video(x[0], output, fps)
    print(f"Saved video: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="outputs/xen-gen1-v")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="outputs/xen-gen1-v/generated.mp4")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args()

    if a.duration is not None:
        if not 1.0 <= a.duration <= 15.0:
            raise ValueError("--duration must be between 1 and 15 seconds")
        frames = max(2, round(a.duration * a.fps))
    else:
        frames = a.frames or 16

    generate(a.model_dir, a.prompt, a.output, frames, a.size, a.steps, a.seed, a.fps)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model.image_model import XENImageModel
from model.image_conditioner import XENImageTextEncoder
from model.tokenizer import XENTokenizer

SYSTEM_PROMPT_PATH = Path("configs/system_prompt_i.txt")


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing system prompt: {SYSTEM_PROMPT_PATH}")
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def cosine_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """Return the same cumulative alpha schedule used by image training."""
    value = ((t.float() / 999.0) + 0.008) / 1.008
    return torch.cos(value * torch.pi / 2).pow(2).clamp(1e-4, 0.9999)


def make_timesteps(num_steps: int, device: torch.device) -> torch.Tensor:
    if num_steps < 1 or num_steps > 1000:
        raise ValueError("--steps must be between 1 and 1000")
    # Descending integer timesteps from 999 to 0.
    return torch.linspace(999, 0, num_steps, device=device).round().long().unique()


@torch.inference_mode()
def generate(
    model,
    conditioner,
    tokenizer,
    prompt: str,
    size: int,
    steps: int,
    seed: int,
    device: torch.device,
):
    if size % 8 != 0:
        raise ValueError("--size must be divisible by 8")

    system_prompt = load_system_prompt()
    text = system_prompt + "\n\nUSER VISUAL REQUEST:\n" + prompt.strip()
    ids = torch.tensor(
        tokenizer.encode(text, max_length=256),
        dtype=torch.long,
        device=device,
    )[None, :]

    text_embedding = conditioner(ids)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        (1, 3, size, size),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    timesteps = make_timesteps(steps, device)
    for index, timestep in enumerate(timesteps):
        t = timestep.expand(1)
        pred_noise = model(x, t, text_embedding)

        alpha_t = cosine_alpha_bar(t).view(1, 1, 1, 1)
        x0 = (x - (1.0 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        x0 = x0.clamp(-1.5, 1.5)

        if index == len(timesteps) - 1:
            x = x0
            break

        next_t = timesteps[index + 1].expand(1)
        alpha_next = cosine_alpha_bar(next_t).view(1, 1, 1, 1)

        # Deterministic DDIM-style update (eta=0).
        direction = (1.0 - alpha_next).clamp_min(0).sqrt() * pred_noise
        x = alpha_next.sqrt() * x0 + direction

    image = ((x[0].clamp(-1, 1) + 1.0) * 127.5).round().byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate an image with XEN-GEN1-I.")
    parser.add_argument(
        "--model-dir",
        default="outputs/xen-gen1-i",
        help="Directory containing model.pt and tokenizer.json",
    )
    parser.add_argument("--prompt", required=True, help="Text description of the image")
    parser.add_argument("--output", default="outputs/xen-gen1-i/generated.png")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "model.pt"
    tokenizer_path = model_dir / "tokenizer.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Train XEN-GEN1-I first."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer: {tokenizer_path}. Train XEN-GEN1-I first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = XENImageModel().to(device)
    conditioner = XENImageTextEncoder().to(device)
    model.load_state_dict(checkpoint["model"])
    conditioner.load_state_dict(checkpoint["conditioner"])
    model.eval()
    conditioner.eval()

    tokenizer = XENTokenizer.load(tokenizer_path)

    image = generate(
        model=model,
        conditioner=conditioner,
        tokenizer=tokenizer,
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

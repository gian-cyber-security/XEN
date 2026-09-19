# XEN

> **THIS AI JUST RUN IN LOCAL OR IN HUGGINGFACE!**

**XEN** is a from-scratch AI model project with separate model families for text, image, and future video generation.

- **XEN-GEN1-T** → text, coding, reasoning, and math.
- **XEN-GEN1-I** → image generation.
- **XEN-GEN1-V** → video generation (future model).

XEN does not load Qwen, Llama, Mistral, Gemma, or another pretrained language model for XEN-GEN1-T. Its tokenizer, architecture, and weights are created by this repository and initialized from scratch.

## Current architecture

### XEN-GEN1-T

- Decoder-only causal Transformer
- Byte-level UTF-8 tokenizer
- RoPE positional encoding
- RMSNorm
- Multi-head causal self-attention
- SwiGLU feed-forward blocks
- Tied input/output embeddings
- Autoregressive next-token prediction
- Random initialization

### XEN-GEN1-I

- From-scratch conditional diffusion baseline
- Dedicated image text conditioner
- U-Net-style image denoising architecture
- 256×256 baseline resolution
- Text-to-image training
- DDIM-style deterministic local sampler
- Separate image system prompt

The default development target is an **RTX 4060 8GB + 32GB RAM** machine. XEN is a research prototype; capability depends heavily on dataset quality, training compute, and training duration.

## Install

~~~bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
~~~

## XEN-GEN1-T

### Dataset

Create `datasets/train.jsonl`:

~~~json
{"instruction":"What is 2 + 2?","response":"4"}
{"instruction":"Explain what a variable is in programming.","response":"A variable is a named place for storing a value that a program can read or change."}
~~~

### Train

~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen
~~~

Resume training:

~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen --resume outputs/xen/latest.pt
~~~

### Generate text

~~~bash
python inference/generate.py --model-dir outputs/xen --prompt "What is 2 + 2?"
~~~

## XEN-GEN1-I — Image Generation

XEN-GEN1-I is trained separately from XEN-GEN1-T.

### 1. Prepare the image dataset

Create `datasets/image_data.jsonl`:

~~~json
{"image":"datasets/images/cat.jpg","caption":"a small orange cat sitting in a garden"}
{"image":"datasets/images/castle.jpg","caption":"a fantasy castle on a floating island under dramatic clouds"}
~~~

Each entry contains:

- `image` — path to the training image.
- `caption` — text description of the image.

Use a diverse, clean dataset with accurate captions.

### 2. Train XEN-GEN1-I

Run:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i
~~~

For a custom run:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i --steps 10000 --batch-size 2 --lr 0.0002
~~~

Training creates:

~~~text
outputs/xen-gen1-i/
├── model.pt
└── tokenizer.json
~~~

### 3. Generate an image locally

After training has completed, run:

~~~bash
python inference/image_generate.py --model-dir outputs/xen-gen1-i --prompt "a small orange cat sitting in a quiet garden, soft morning light"
~~~

The generated image is saved to:

~~~text
outputs/xen-gen1-i/generated.png
~~~

Choose a different output file with:

~~~bash
python inference/image_generate.py --model-dir outputs/xen-gen1-i --prompt "a low-poly fantasy castle on a floating island" --output outputs/my_image.png
~~~

### 4. Generation settings

The default generation settings are:

- `--size 256` — output resolution.
- `--steps 50` — sampling steps.
- `--seed 42` — random seed.

Example:

~~~bash
python inference/image_generate.py --model-dir outputs/xen-gen1-i --prompt "a futuristic city at night" --size 256 --steps 50 --seed 123
~~~

For reproducible results, use the same prompt and seed.

### 5. CPU or NVIDIA GPU

The script automatically uses CUDA when available and otherwise falls back to CPU.

For an NVIDIA GPU:

~~~text
XEN-GEN1-I
    ↓
PyTorch CUDA
    ↓
RTX GPU
    ↓
generated PNG
~~~

CPU generation is supported but can be significantly slower.

### 6. What to expect

XEN-GEN1-I is a small model trained entirely from scratch. It is not a pretrained large image generator. Early generations may contain noise, distorted shapes, weak prompt understanding, or inconsistent composition.

If the output is poor, check:

- training loss;
- number of training steps;
- dataset size and diversity;
- caption quality;
- checkpoint integrity;
- sampling steps.

The current model is a development baseline. Improving the dataset, text conditioning, diffusion schedule, and model capacity can improve future generations.


## XEN-GEN1-V — Video Generation

XEN-GEN1-V is the separate video-generation family. It is implemented as a compact conditional video diffusion model trained from scratch.

### Architecture

- Factorized 3D residual U-Net-style denoiser.
- Separate spatial and temporal convolutions so motion information is processed across frames.
- Trainable byte-level text conditioner.
- Temporal length is not hard-coded in the model; the same checkpoint can accept different clip lengths subject to available memory.
- Initial development target: 16 frames at 128×128 on an RTX 4060 8GB.
- The generation interface supports a requested duration from 1 to 15 seconds.

This is a research/development model, not a pretrained Sora/Veo-class system. Video quality depends strongly on dataset quality, training duration, model capacity, and available compute.

### Dataset

Create `datasets/video_data.jsonl` using:

~~~json
{"video":"datasets/videos/cat.mp4","caption":"a small orange cat walking through a quiet garden, soft morning light"}
{"video":"datasets/videos/car.mp4","caption":"a red sports car driving through a city at night, cinematic tracking camera"}
~~~

Each entry contains:

- `video` — path to the training video.
- `caption` — accurate visual and temporal description.

The training loader samples a temporal clip from each source video, resizes it, normalizes it to [-1, 1], and trains the model to predict diffusion noise.

### Train XEN-GEN1-V

Default development run:

~~~bash
python training/video_train.py --data datasets/video_data.jsonl --output outputs/xen-gen1-v
~~~

RTX 4060-oriented starting point:

~~~bash
python training/video_train.py --data datasets/video_data.jsonl --output outputs/xen-gen1-v --frames 16 --size 128 --batch-size 1 --steps 10000 --grad-accumulation 8
~~~

The training checkpoint contains the video denoiser, text conditioner, and tokenizer.

### Generate a video locally

Generate a short clip:

~~~bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a red ball rolling across a wooden table"
~~~

Choose a duration between 1 and 15 seconds:

~~~bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a red ball rolling across a wooden table" --duration 5 --fps 8
~~~

The output defaults to:

~~~text
outputs/xen-gen1-v/generated.mp4
~~~

Available generation controls:

- `--duration 1..15` — requested video duration in seconds.
- `--frames` — directly choose frame count instead of duration.
- `--size` — square output resolution.
- `--steps` — diffusion sampling steps.
- `--fps` — output frames per second.
- `--seed` — reproducible random seed.

Longer durations and higher resolutions can require substantially more VRAM. The duration option exposes 1–15 seconds, but the actual practical limit depends on the user's GPU/RAM and the trained checkpoint. CPU generation is supported but can be extremely slow.

### Local and Hugging Face

The model is designed so the checkpoint and inference code can be used locally or packaged for Hugging Face. Hugging Face deployment should use a GPU Space for practical generation.

Image-to-video conditioning is intentionally reserved for a future XEN-GEN1-V revision; the current GEN1 checkpoint is text-to-video.

## API

Start the XEN API:

~~~bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
~~~

## Development

Improve dataset quality and training infrastructure before scaling model size. More examples do not automatically make a small model proportionally smarter; model capacity, token count, data quality, and optimization all matter.

## License

MIT

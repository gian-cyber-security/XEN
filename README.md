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

### 1. Prepare image dataset

Create a JSONL file such as `datasets/image_data.jsonl`:

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

For a custom training run:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i --steps 10000 --batch-size 2 --lr 0.0002
~~~

Wait until the model has produced a trained checkpoint.

### 3. Generate an image

**Important:** the image inference script is not currently included in this repository. Therefore, there is no working image-generation command yet.

The intended workflow is:

~~~text
image_data.jsonl
       ↓
training/image_train.py
       ↓
XEN-GEN1-I checkpoint
       ↓
inference/image_generate.py
       ↓
generated PNG
~~~

Once `inference/image_generate.py` is added, the README will be updated with the exact command, prompt format, output path, and optional generation settings.

**Do not create or guess an inference command yourself.** The command will depend on the final checkpoint format and sampler implemented by the repository.

### What to expect from the first model

XEN-GEN1-I is a small model trained from scratch, not a pretrained large image generator. Early generations may contain noise, distorted shapes, weak prompt understanding, or inconsistent composition. This is expected during early development.

If generation quality is poor, inspect:

- training loss;
- number of training steps;
- dataset size and diversity;
- caption quality;
- checkpoint integrity;
- inference sampler and noise schedule.

## API

Start the XEN API:

~~~bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
~~~

## Development

Improve dataset quality and training infrastructure before scaling model size. More examples do not automatically make a small model proportionally smarter; model capacity, token count, data quality, and optimization all matter.

## License

MIT

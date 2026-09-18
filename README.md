# XEN

**XEN** is a from-scratch Text-to-Text language model project designed for local experimentation and iterative training.

XEN does not load Qwen, Llama, Mistral, Gemma, or another pretrained language model. Its tokenizer, architecture, and weights are created by this repository and initialized from scratch.

## Current architecture

- Decoder-only causal Transformer
- Byte-level UTF-8 tokenizer (no normal-language OOV problem)
- RoPE positional encoding
- RMSNorm
- Multi-head causal self-attention using PyTorch scaled dot-product attention
- SwiGLU feed-forward blocks
- Tied input/output embeddings
- Autoregressive next-token prediction
- Random initialization

The default training configuration is aimed at an **RTX 4060 8GB + 32GB RAM** development machine. It is still a research prototype; capability depends primarily on dataset quality, training compute, and training duration.

## Install

~~~bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
~~~

## Dataset

Create `datasets/train.jsonl` with one JSON object per line:

~~~json
{"instruction":"What is 2 + 2?","response":"4"}
{"instruction":"Explain what a variable is in programming.","response":"A variable is a named place for storing a value that a program can read or change."}
~~~

For a useful model, use a large, diverse, cleaned dataset rather than a tiny collection of examples.

## Train

The training engine reads `configs/train.yaml` and supports batching, validation split, CUDA mixed precision, gradient accumulation, clipping, cosine learning-rate decay with warmup, periodic checkpoints, and resume training.

~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen
~~~

Resume from a checkpoint:

~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen --resume outputs/xen/latest.pt
~~~

The best validation model is saved as `outputs/xen/model.pt`. Training state is saved as `outputs/xen/latest.pt` and periodic `checkpoint-*.pt` files.

## Generate

~~~bash
python inference/generate.py --model-dir outputs/xen --prompt "What is 2 + 2?"
~~~

## API

~~~bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
~~~

## Scaling XEN

A practical development path is to improve data quality and training infrastructure first, then scale model size when the available GPU allows it. More examples do not automatically make a small model proportionally smarter; model capacity, token count, data quality, and optimization all matter.

## License

MIT

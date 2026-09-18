# XEN

**XEN** is a Python-first, from-scratch Text-to-Text language model project.

XEN does not load Qwen, Llama, Mistral, Gemma, or another pretrained language model. The architecture, tokenizer, and weights are created by this repository and initialized from scratch.

This first prototype is intentionally small for experimental training on a normal GPU. It is a research prototype, not a claim of frontier-level capability.

## Architecture

- Decoder-only causal Transformer
- Learned token and positional embeddings
- Causal self-attention
- Feed-forward MLP blocks
- Layer normalization
- Tied input/output embeddings
- Autoregressive next-token prediction

## Python 100%

All project source code is Python. No C++, Rust, TypeScript, or pretrained language model is part of XEN.

## Train

Create datasets/train.jsonl:

~~~json
{"instruction":"What is 2 + 2?","response":"4"}
{"instruction":"Say hello.","response":"Hello!"}
~~~

Then run:

~~~bash
python training/train.py --data datasets/train.jsonl --output outputs/xen
~~~

## Generate

~~~bash
python inference/generate.py --model-dir outputs/xen --prompt "What is 2 + 2?"
~~~

## API

~~~bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
~~~

The /health endpoint reports XEN-from-scratch.

## Next

The prototype can later grow with a stronger tokenizer, larger context, RoPE, RMSNorm, improved attention, larger datasets, distributed training, evaluation, and larger XEN model sizes.

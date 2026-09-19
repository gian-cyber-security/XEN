# XEN

> **THIS AI JUST RUN IN LOCAL OR IN HUGGINGFACE!**

XEN is the **collection/hub repository** for the XEN model family. Each model generation now lives in its own repository so training, inference, releases, and documentation can be managed independently.

## Models

| Model | Purpose | Repository |
|---|---|---|
| **XEN-GEN1-T** | Text, coding, reasoning, math | https://github.com/gian-cyber-security/xen-gen1-t |
| **XEN-GEN1-I** | Text-to-image and future image conditioning | https://github.com/gian-cyber-security/xen-gen1-i |
| **XEN-GEN1-V** | Text-to-video, 1–15 second interface | https://github.com/gian-cyber-security/xen-gen1-v |

## Family design

XEN keeps text, image, and video as separate model families instead of combining everything into one multimodal model.

- **GEN1-T** → language intelligence.
- **GEN1-I** → image generation.
- **GEN1-V** → video generation.
- Future generations can use the same naming scheme: GEN2-T, GEN2-I, GEN2-V, and so on.

## Web search

The text-model family supports an optional web-search layer through the `ddgs` metasearch package. The text generators can automatically search for time-sensitive prompts and can also be forced with `--web-search` or disabled with `--no-web-search`. Search results are supplied as context to the local model; the model weights remain local.

The web-search layer is currently integrated into **GEN1-T**, **GEN1-T-Code**, **GEN1-T-Fast**, and **GEN1-T-Plan**. GEN1-I and GEN1-V remain generation-focused models and do not use the text-answer search layer.

## Deployment

The individual model repositories contain their own training and local inference instructions. The models are intended for local use and can be packaged for Hugging Face deployment.

## Hardware

The current development baseline is an RTX 4060 8GB + 32GB RAM system. Actual capability depends on model size, dataset, training duration, resolution, clip length, and available compute.

## License

MIT

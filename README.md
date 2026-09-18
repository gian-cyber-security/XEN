# XEN

**XEN** is an open, modular AI system focused on text-to-text generation, instruction following, reasoning, coding, and mathematics.

## Text-to-Text foundation

- Hugging Face Transformers inference
- Configurable open-weight base model
- Chat/instruction formatting
- Local CLI inference
- FastAPI server
- JSONL dataset preparation
- LoRA/QLoRA supervised fine-tuning
- Basic generation evaluation
- Configurable VRAM/training settings

## Structure

~~~text
XEN/
├── api/server.py
├── configs/train.yaml
├── datasets/README.md
├── evaluation/evaluate.py
├── evaluation/benchmarks.jsonl
├── inference/generate.py
├── model/loader.py
├── scripts/prepare_dataset.py
├── training/train.py
├── requirements.txt
└── README.md
~~~

## Install

Python 3.10+ recommended.

~~~bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
~~~

## Run

~~~bash
python inference/generate.py --model Qwen/Qwen3-4B-Instruct-2507 --prompt "Explain photosynthesis simply."
~~~

The first run downloads the selected model from Hugging Face.

## API

~~~bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
~~~

POST JSON to /generate with prompt, max_new_tokens, temperature, and top_p.

## Fine-tuning

Training uses supervised fine-tuning of an existing open-weight base model. It is not pretending to train a frontier foundation model from zero.

Example JSONL:

~~~json
{"instruction":"What is 2 + 2?","response":"2 + 2 = 4."}
~~~

~~~bash
python scripts/prepare_dataset.py --input datasets/train.jsonl --output datasets/processed
python training/train.py --config configs/train.yaml
~~~

Set load_in_4bit=true for QLoRA on supported hardware. Batch size, sequence length, LoRA rank, and other settings are configurable.

## Data quality

Training data should be accurate, diverse, appropriately licensed, deduplicated, and free of API keys, passwords, tokens, and unnecessary personal information. Do not automatically train the production model on every conversation. Prefer an opt-in collection -> redaction -> filtering -> deduplication -> evaluation -> fine-tuning -> benchmark -> deployment pipeline.

## Evaluation

~~~bash
python evaluation/evaluate.py --model ./outputs/xen --data evaluation/benchmarks.jsonl
~~~

## Roadmap

- [x] XEN Text-to-Text foundation
- [x] Local inference
- [x] API inference
- [x] Dataset preparation
- [x] LoRA/QLoRA training pipeline
- [x] Basic evaluation
- [ ] XEN-Code
- [ ] XEN-Reason
- [ ] XEN-Vision
- [ ] XEN-Image
- [ ] XEN-ImageEdit
- [ ] XEN-Video

## License

MIT. Check every base model and dataset license before redistribution or commercial use.

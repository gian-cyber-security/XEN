# XEN-GEN1-I — Image Generation Guide

XEN-GEN1-I is the image generation model in the XEN family, separate from XEN-GEN1-T.

> **THIS AI JUST RUN IN LOCAL OR IN HUGGINGFACE!**

## Status

XEN-GEN1-I is currently a from-scratch baseline image model. Image quality depends heavily on the training dataset and training progress.

**Important:** do not attempt image generation until a compatible image-model checkpoint is available.

## 1. Install dependencies

From the XEN repository root:

~~~bash
pip install -r requirements.txt
~~~

Using a dedicated Python virtual environment is recommended.

## 2. Prepare the dataset

Training uses JSONL with this format:

~~~json
{"image":"datasets/images/cat.jpg","caption":"a small orange cat sitting in a garden"}
{"image":"datasets/images/car.jpg","caption":"a low poly red sports car on a mountain road"}
~~~

Each line contains:

- `image`: path to the image file.
- `caption`: description of the image.

Use a diverse dataset with clear, descriptive captions. The training dataset must exist before starting image training.

## 3. Train XEN-GEN1-I

Run:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i
~~~

You can also specify the number of training steps:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i --steps 10000 --batch-size 2 --lr 0.0002
~~~

Training takes time, and early results may be very simple, unstable, or noisy. This is expected for a model trained entirely from scratch.

## 4. Generate images after training

Once a compatible checkpoint exists, an image inference script can be used to generate images.

**Current repository status:** `inference/image_generate.py` has not been added yet. Do not use a generation command that assumes this script exists.

The current workflow is:

~~~text
Dataset
  ↓
training/image_train.py
  ↓
XEN-GEN1-I checkpoint
  ↓
image inference script (to be added)
  ↓
PNG output
~~~

The future inference script should accept a text prompt, load the XEN-GEN1-I checkpoint, tokenizer, and image system prompt, then save the generated image as a PNG file.

## 5. Example prompts

Example prompt:

~~~text
a small orange cat sitting in a quiet garden, soft morning light, detailed leaves, natural composition
~~~

Another example:

~~~text
a low-poly fantasy castle on a floating island, mountains in the distance, dramatic clouds, cinematic lighting
~~~

## 6. Troubleshooting

### The output is noise or random shapes

Possible causes include:

- the model has not been trained for enough steps;
- the dataset is too small;
- captions are unclear or inconsistent;
- the checkpoint is incorrect or incomplete;
- the inference sampler or noise schedule does not match the training setup.

### The model does not understand prompts well

XEN-GEN1-I currently uses a simple text conditioner. Its prompt understanding is expected to be much weaker than large image-generation systems that use stronger text encoders and cross-attention.

### Out of VRAM

With an RTX 4060 8 GB, try:

- lowering the batch size;
- using gradient accumulation;
- reducing training resolution if supported by the training pipeline;
- closing other applications that are using the GPU.

## 7. Development notes

Do not evaluate the model from a single generated image. Keep a fixed set of test prompts and compare outputs after changes to the architecture, dataset, or training configuration.

XEN-GEN1-I is a separate model from XEN-GEN1-T:

- **XEN-GEN1-T** → text, coding, reasoning, and math.
- **XEN-GEN1-I** → text-to-image and image-to-image/editing.
- **XEN-GEN1-V** → video generation (future model).

`configs/system_prompt_i.txt` contains the dedicated system prompt for XEN-GEN1-I and is separate from the XEN-GEN1-T system prompt.

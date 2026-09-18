# XEN-GEN1-I — Image Generation Guide

XEN-GEN1-I adalah model image generation XEN yang dipisahkan dari XEN-GEN1-T.

> **THIS AI JUST RUN IN LOCAL OR IN HUGGINGFACE!**

## Status

XEN-GEN1-I saat ini adalah baseline image model yang dilatih dari nol. Kualitas gambar sangat bergantung pada dataset dan lama training.

**Penting:** jangan langsung mencoba generate sebelum checkpoint model image sudah tersedia.

## 1. Install dependencies

Dari root repository XEN:

~~~bash
pip install -r requirements.txt
~~~

Disarankan menggunakan environment Python terpisah, misalnya virtual environment.

## 2. Siapkan dataset

Training menggunakan JSONL dengan format:

~~~json
{"image":"datasets/images/cat.jpg","caption":"a small orange cat sitting in a garden"}
{"image":"datasets/images/car.jpg","caption":"a low poly red sports car on a mountain road"}
~~~

Setiap baris berisi:

- `image`: path ke file gambar.
- `caption`: deskripsi gambar.

Gunakan dataset yang beragam dan caption yang jelas. Dataset training harus sudah tersedia sebelum menjalankan image training.

## 3. Train XEN-GEN1-I

Jalankan:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i
~~~

Contoh jika ingin mengatur jumlah step:

~~~bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i --steps 10000 --batch-size 2 --lr 0.0002
~~~

Training membutuhkan waktu dan hasil pertama kemungkinan masih sederhana atau belum stabil. Itu normal untuk model yang benar-benar dilatih dari nol.

## 4. Generate gambar setelah training

Setelah ada checkpoint model yang kompatibel, jalankan inference image generator.

**Catatan penting:** repository saat ini belum menyediakan script inference `inference/image_generate.py`. Jadi jangan mengarang command generate yang belum tersedia.

Workflow yang benar untuk saat ini:

~~~text
Dataset
  ↓
training/image_train.py
  ↓
checkpoint XEN-GEN1-I
  ↓
inference script (akan ditambahkan)
  ↓
PNG hasil generate
~~~

Inference script akan menerima prompt teks, memuat checkpoint XEN-GEN1-I + tokenizer + system prompt image, lalu menyimpan hasil sebagai file PNG.

## 5. Contoh prompt

Contoh prompt yang bisa digunakan setelah inference script tersedia:

~~~text
a small orange cat sitting in a quiet garden, soft morning light, detailed leaves, natural composition
~~~

Atau:

~~~text
a low-poly fantasy castle on a floating island, mountains in the distance, dramatic clouds, cinematic lighting
~~~

## 6. Troubleshooting

### Hasil berupa noise atau bentuk random

Kemungkinan penyebab:

- model belum cukup lama dilatih;
- dataset terlalu sedikit;
- caption kurang jelas;
- checkpoint belum benar;
- sampler/schedule inference belum sesuai dengan training.

### Model tidak memahami prompt dengan baik

XEN-GEN1-I saat ini memakai text conditioner sederhana. Kemampuan memahami prompt belum setara model image generator besar yang memakai text encoder dan cross-attention yang lebih kuat.

### VRAM habis

Dengan RTX 4060 8 GB, coba:

- turunkan batch size;
- gunakan gradient accumulation;
- kurangi resolusi training jika pipeline yang digunakan mendukungnya;
- pastikan proses lain yang memakai GPU sudah ditutup.

## 7. Prinsip pengembangan

Jangan menilai model hanya dari satu gambar. Simpan beberapa prompt tetap dan bandingkan hasil setelah perubahan arsitektur, dataset, atau training.

XEN-GEN1-I adalah model terpisah dari XEN-GEN1-T:

- **XEN-GEN1-T** → text, coding, reasoning, math.
- **XEN-GEN1-I** → text-to-image dan image-to-image/editing.
- **XEN-GEN1-V** → video generation (future model).

`configs/system_prompt_i.txt` adalah system prompt khusus XEN-GEN1-I dan tidak sama dengan system prompt XEN-GEN1-T.

# 🦥 Kaggle Unsloth Notebooks — T4×2 GPU Edition

> **Production-ready fine-tuning & quantization notebooks for LLMs on Kaggle's free T4×2 GPU environment, powered by [Unsloth](https://github.com/unslothai/unsloth).**

[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/aab20abdullah)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4%20×2-76B900?logo=nvidia&logoColor=white)](https://www.kaggle.com/docs/efficient-gpu-usage)

---

## 📌 Overview

This repository contains a curated collection of Jupyter notebooks specifically engineered to run **fine-tuning, QLoRA training, and GGUF quantization** workflows on **Kaggle's free T4×2 dual-GPU environment** using the Unsloth library.

All notebooks are pre-adapted for Kaggle's constraints (disk limits, RAM limits, internet toggle) and follow a complete **train → quantize → publish** pipeline targeting Hugging Face as the deployment destination.

**Target models include:** Llama-3, Llama-3.2, DeepSeek-7B, and other popular open-weight LLMs.

---

## 📁 Repository Structure

```
kaggle-UNSLOTH-NOTEBOOKS/
└── kaggle_T4x2_converted/       # Notebooks converted & optimized for T4×2 dual-GPU
    ├── *.ipynb                  # Fine-tuning, QLoRA, GGUF quantization notebooks
    └── ...
```

> All notebooks in `kaggle_T4x2_converted/` are production-tested on Kaggle's T4×2 accelerator and require **no local GPU**.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🚀 **2× Faster Training** | Unsloth's custom Triton kernels accelerate training vs. standard HF Trainer |
| 💾 **70% Less VRAM** | QLoRA + 4-bit loading allows large models to fit in 15GB per GPU |
| 🔁 **Dual-GPU Support** | Device mapping and pipeline parallelism across T4×2 |
| 📦 **GGUF Quantization** | Q2_K_M, Q4_K_M, Q8_0 and more via integrated `llama.cpp` conversion |
| ☁️ **HF Auto-Push** | Models and adapters pushed directly to Hugging Face Hub post-training |
| 🆓 **100% Free Compute** | Designed for Kaggle's free tier — no paid cloud required |

---

## 🛠️ Workflows Covered

### 1. 🎯 QLoRA Fine-Tuning
Fine-tune large language models using Parameter-Efficient Fine-Tuning (PEFT) with `r=16` LoRA adapters on custom datasets (Alpaca format, conversational, etc.).

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### 2. ⚡ GGUF Quantization & Export
Convert fine-tuned adapters to GGUF format for local inference with `llama.cpp`, `Ollama`, or `LM Studio`.

```python
# Export to multiple quantization levels
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
model.push_to_hub_gguf("aab20abdullah/my-model-gguf", tokenizer, token="hf_...")
```

### 3. 🤗 Push to Hugging Face Hub
All notebooks include automated publishing of both LoRA adapters and merged/quantized models to `https://huggingface.co/aab20abdullah`.

---

## ⚙️ Setup & Usage

### Step 1 — Open in Kaggle
Click any `.ipynb` from the `kaggle_T4x2_converted/` folder and import it directly into Kaggle.

> **Kaggle → Code → New Notebook → File → Import Notebook**

### Step 2 — Enable T4×2 Accelerator
```
Settings → Accelerator → GPU T4 × 2
Settings → Internet → On (required for pip installs & HF push)
```

### Step 3 — Install Dependencies (first cell)
```bash
%%capture
!pip install pip3-autoremove
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
import os
os.environ["WANDB_DISABLED"] = "true"
```

### Step 4 — Run All Cells
All notebooks are self-contained and runnable end-to-end. No external dataset downloads required for demo notebooks.

---

## 📊 Hardware Requirements

| Resource | Value |
|---|---|
| GPU | NVIDIA T4 × 2 (30GB total VRAM) |
| RAM | ~13GB (Kaggle default) |
| Disk | ~19GB (Kaggle default) |
| Runtime | Up to 30h / week (Kaggle free tier) |

> ⚠️ **Disk Management:** GGUF conversion requires careful staging. Notebooks include disk-check utilities and cleanup steps to avoid OOM/disk-full errors during quantization.

---

## 🤖 Models & Published Outputs

Fine-tuned and quantized models from these notebooks are published to:

**🤗 [huggingface.co/aab20abdullah](https://huggingface.co/aab20abdullah)**

| Model Base | Task | Format |
|---|---|---|
| Llama-3.2 | Domain fine-tuning (chess, Turkic languages) | LoRA + GGUF |
| DeepSeek-7B | General instruction tuning | Q2_K_M GGUF |
| Llama-3 | Custom dataset SFT | LoRA adapter |

---

## 🔧 Troubleshooting

**`No module named 'unsloth'`**
→ Reinstall using the Kaggle-specific install command in Step 3 above.

**Disk full during GGUF conversion**
→ Run `!df -h` to check space. Delete intermediate checkpoints before converting:
```bash
!rm -rf /kaggle/working/model_checkpoints/checkpoint-*
```

**CUDA OOM on T4×2**
→ Reduce `per_device_train_batch_size` to `1` and enable `gradient_checkpointing = "unsloth"`.

**Push to Hub fails after GGUF conversion**
→ Use `model.push_to_hub_gguf()` with `token=` argument explicitly. Avoid large shard sizes by setting `maximum_memory_usage=0.7`.

---

## 📚 References

- [Unsloth Official Repo](https://github.com/unslothai/unsloth)
- [Unsloth Kaggle Notebooks](https://github.com/unslothai/notebooks)
- [Kaggle GPU Docs](https://www.kaggle.com/docs/efficient-gpu-usage)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [llama.cpp GGUF Formats](https://github.com/ggerganov/llama.cpp)

---

## 👤 Author

**Abdullah A.B.**
- 🤗 Hugging Face: [@aab20abdullah](https://huggingface.co/aab20abdullah)
- 🐙 GitHub: [@AAB20](https://github.com/AAB20)
- 📸 Instagram: [@abbdr4](https://www.instagram.com/abbdr4/)

AI/ML Engineer · Certified Trainer · LLM Fine-Tuning & Quantization Specialist

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built with 🦥 Unsloth + ☕ free Kaggle GPUs</i>
</p>

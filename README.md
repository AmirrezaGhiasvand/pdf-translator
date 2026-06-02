# PDF Translator — English to Persian (EN→FA)

A fine-tuned LLM pipeline for translating English PDF documents to Persian. This project fine-tunes **Qwen2.5-0.5B-Instruct** using **QLoRA** on a parallel EN↔FA dataset, turning a general-purpose language model into a focused translation model.

---

## Motivation

General-purpose small LLMs produce poor translation quality out of the box, especially for Persian. Rather than using a pre-built translation model, this project demonstrates the full fine-tuning pipeline — from raw data to a deployable model — as a learning exercise in applied ML.

---

## Approach

| Component | Choice | Reason |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Modern architecture, multilingual-aware, fits in 4GB VRAM |
| Fine-tuning method | QLoRA | Trains only ~1M parameters, feasible on consumer GPU |
| Dataset | OPUS-100 EN↔FA | 1M parallel sentence pairs, freely available |
| Training library | TRL + PEFT | Industry standard for instruction fine-tuning |

---

## Project Structure

```
pdf-translator/
├── data/
│   └── processed/        # formatted train.json and val.json (not tracked by git)
├── src/
│   ├── data_prep.py      # downloads and formats the dataset
│   ├── train.py          # QLoRA fine-tuning script
│   └── inference.py      # compare base vs fine-tuned model
├── models/               # saved LoRA adapter after training (not tracked by git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

| Component | Minimum | Used in this project |
|---|---|---|
| GPU | 4GB VRAM | NVIDIA GTX 1630 (4GB) |
| RAM | 8GB | - |
| Storage | 10GB free | ~7GB (model + dataset) |
| Python | 3.11 | 3.11 |
| CUDA | 11.8+ | 12.7 |

> CPU-only training is possible but will take significantly longer (days vs hours).

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/AmirrezaGhiasvand/pdf-translator.git
cd pdf-translator
```

### 2. Create virtual environment with Python 3.11
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA support
```bash
# CUDA 12.4 (compatible with CUDA 12.x drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch
```

### 5. Set UTF-8 encoding (Windows only)
```bash
set PYTHONUTF8=1
```

---

## Usage

### Step 1: Prepare the dataset
```bash
python src/data_prep.py
```
Downloads 50k EN↔FA sentence pairs from OPUS-100 and saves them to `data/processed/`.

### Step 2: Train the model
```bash
python src/train.py
```

**Test run** (default — ~30 minutes on GTX 1630):
```python
EPOCHS            = 1
MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES   = 500
```

**Full training** (~9-15 hours on GTX 1630):
```python
EPOCHS            = 3
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None
```

### Step 3: Test the model
```bash
python src/inference.py
```
Runs a side-by-side comparison of the base model vs fine-tuned model on sample sentences.

---

## How QLoRA Works

Standard fine-tuning updates all 500M parameters — requiring 24GB+ VRAM. QLoRA makes this feasible on consumer hardware through two techniques:

**Quantization:** The base model is loaded in 4-bit precision, reducing VRAM from ~2GB to ~500MB.

**LoRA (Low-Rank Adaptation):** Instead of updating all parameters, two small matrices are injected into the attention layers. Only ~1M parameters are trained instead of 500M.

```
Base model (frozen, 4-bit)     ~500MB VRAM
LoRA adapter (trained)          ~50MB VRAM
──────────────────────────────────────────
Total trainable parameters:     0.22% of model
```

After training, the adapter is merged back into the base model.

---

## Results

> ⚠️ Full training in progress. Results will be updated after completion.

### Test Run (5k samples, 1 epoch)

| Sentence | Base Model | Fine-tuned |
|---|---|---|
| Hello, how are you? | سلام، چه سوالی بفرستید؟ + hallucinations | باشید، ما بهمچون که؟ |
| Please submit your report by Friday. | Repeats endlessly | اين ميشه بودن که اون پیروش به روز گذاريم |

**Observations:**
- Base model translates partially but hallucinates, adds hashtags, and loops
- Fine-tuned model learned to stop generating at the right point ✅
- Translation quality needs full training to improve

### Full Training Results
```
BLEU score (base model):     TBD
BLEU score (fine-tuned):     TBD
```

---

## Roadmap

- [x] Dataset preparation
- [x] QLoRA training pipeline
- [x] Inference and comparison script
- [ ] Full training run (3 epochs, 45k samples)
- [ ] BLEU score evaluation
- [ ] Merge LoRA adapter into base model
- [ ] Convert to GGUF for Ollama
- [ ] Integrate into PDF translation pipeline

---

## Dependencies

| Package | Purpose |
|---|---|
| transformers | Model loading and tokenization |
| peft | LoRA adapter creation and training |
| trl | SFTTrainer for instruction fine-tuning |
| bitsandbytes | 4-bit quantization |
| datasets | OPUS dataset loading |
| accelerate | Distributed training utilities |
| pypdf | PDF text extraction |
| sacrebleu | BLEU score evaluation |

---

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [OPUS-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

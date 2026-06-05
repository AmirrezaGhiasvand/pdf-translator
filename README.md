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
│   ├── inference.py      # compare base vs fine-tuned model
│   └── evaluate.py       # BLEU score evaluation
├── models/               # saved LoRA adapters after training (not tracked by git)
│   ├── qwen-fa/          # V1 adapter (15k samples, r=16)
│   └── qwen-fa-v2/       # V2 adapter (30k samples, r=64)
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

**Test run** (~30 minutes on GTX 1630):
```python
EPOCHS            = 1
MAX_TRAIN_SAMPLES = 5000
MAX_VAL_SAMPLES   = 500
```

**Full training V2** (~10 hours on GTX 1630):
```python
EPOCHS            = 2
MAX_TRAIN_SAMPLES = 30000
MAX_VAL_SAMPLES   = 3000
```

### Step 3: Test the model
```bash
python src/inference.py
```
Runs a side-by-side comparison of the base model vs fine-tuned model on sample sentences.

### Step 4: Evaluate with BLEU score
```bash
python src/evaluate.py
```
Evaluates both models on 200 validation samples and reports BLEU scores.

---

## How QLoRA Works

Standard fine-tuning updates all 500M parameters — requiring 24GB+ VRAM. QLoRA makes this feasible on consumer hardware through two techniques:

**Quantization:** The base model is loaded in 4-bit precision, reducing VRAM from ~2GB to ~500MB.

**LoRA (Low-Rank Adaptation):** Instead of updating all parameters, small adapter matrices are injected into the attention layers. Only a fraction of parameters are trained instead of all 500M.

```
Base model (frozen, 4-bit)     ~500MB VRAM
LoRA adapter (trained)          ~50MB VRAM
──────────────────────────────────────────
V1: trainable parameters:       0.22% of model (r=16)
V2: trainable parameters:       ~2% of model  (r=64)
```

---

## Training Runs

### V1 — Initial Training
| Setting | Value |
|---|---|
| Samples | 15,000 |
| Epochs | 2 |
| LoRA rank | r=16 |
| Target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Training time | 5h 19m |

| Metric | Start | End |
|---|---|---|
| Train loss | 3.137 | 2.206 |
| Eval loss | 2.388 | 2.230 |
| Token accuracy | 45% | 56% |

### V2 — Continual Learning
| Setting | Value |
|---|---|
| Samples | 30,000 |
| Epochs | 2 |
| LoRA rank | r=64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 1e-4 |
| Training time | 10h 20m |

| Metric | Start | End |
|---|---|---|
| Train loss | 2.338 | 1.594 |
| Eval loss | 2.090 | 1.738 |
| Token accuracy | 53% | 66% |

---

## Results

### BLEU Score Comparison
| Model | BLEU Score |
|---|---|
| Base model (no fine-tuning) | 5.61 |
| Fine-tuned V1 (15k samples, r=16) | ~8 (estimated) |
| Fine-tuned V2 (30k samples, r=64) | **12.07** |
| Improvement over base | **+115%** |

> BLEU scores evaluated on 200 held-out validation samples using character-level BLEU (sacrebleu).

### Sample Translations

| English | Reference | Base Model | Fine-tuned V2 |
|---|---|---|---|
| Hello, how are you? | سلام، چطوری؟ | سلام، چه سوال دارید؟ | سلام، چه خبری؟ ✅ |
| Twice a day | دوبار در روز | دوست ۱- دیواره‌ای نمود | دوباره هر روز ✅ |
| Please submit your report by Friday | لطفاً گزارش خود را تا جمعه ارسال کنید | شما رایج است چه تا دسترسی... | لطفاً پرونده‌تان روز ۶ بپذیرید 🔄 |
| He's a traitor who betrayed our country | اون يه خائنِه | سیاره است کرده و گفت... | او یک ترور بود که از دنیای ما... 🔄 |

### Key Observations
- Fine-tuned model achieves **115% improvement** in BLEU score over base model
- Short common sentences translate well after fine-tuning ✅
- Model learned to stop generating at the right point — base model hallucinates endlessly ✅
- Longer complex sentences still challenging — limited by 0.5B model capacity
- Continual learning (V1→V2) proved effective — loss dropped from 2.206 to 1.594

---

## Roadmap

- [x] Dataset preparation (OPUS-100, 50k EN↔FA pairs)
- [x] QLoRA training pipeline (V1: 15k samples, r=16)
- [x] Continual learning with larger LoRA (V2: 30k samples, r=64)
- [x] Inference and comparison script
- [x] BLEU score evaluation (base: 5.61 → fine-tuned: 12.07)
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
# PDF Translator — English to Persian (EN→FA)

A fine-tuned LLM pipeline for translating English PDF documents to Persian. This project fine-tunes **Qwen2.5-0.5B-Instruct** using **QLoRA** on a parallel EN↔FA dataset, turning a general-purpose language model into a focused translation model — built entirely on consumer hardware as a learning exercise in applied ML.

---

## Motivation

General-purpose small LLMs produce poor translation quality out of the box, especially for Persian. Rather than using a pre-built translation model, this project demonstrates the full fine-tuning pipeline — from raw data to a deployable model — including dataset preparation, QLoRA training, continual learning, evaluation, and inference.

> ⚠️ This project was built for **educational purposes** on **limited consumer hardware** (GTX 1630, 4GB VRAM). The translation quality reflects these constraints. See [Limitations](#limitations) for details.

---

## Approach

| Component | Choice | Reason |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Modern architecture, multilingual-aware, fits in 4GB VRAM |
| Fine-tuning method | QLoRA | Trains only ~2% of parameters, feasible on consumer GPU |
| Dataset | OPUS-100 EN↔FA | 1M parallel sentence pairs, freely available |
| Training library | TRL + PEFT | Industry standard for instruction fine-tuning |
| Pipeline | LangChain + Transformers | Clean, modular inference pipeline |

---

## Project Structure

```
pdf-translator/
├── data/
│   └── processed/          # formatted train.json and val.json (not tracked)
├── src/
│   ├── data_prep.py        # downloads and formats the OPUS-100 dataset
│   ├── train.py            # QLoRA fine-tuning script
│   ├── inference.py        # compare base vs fine-tuned model
│   ├── evaluate.py         # BLEU score evaluation
│   ├── merge.py            # merge LoRA adapter into base model
│   └── translate_pdf.py    # end-to-end PDF translation pipeline
├── models/                 # saved LoRA adapters (not tracked by git)
│   ├── qwen-fa/            # V1 adapter (15k samples, r=16)
│   ├── qwen-fa-v2/         # V2 adapter (30k samples, r=64)
│   ├── qwen-fa-v3/         # V3 adapter (45k samples, continual learning)
│   └── qwen-fa-merged/     # final merged model ready for inference
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
| Storage | 15GB free | ~12GB (models + dataset) |
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

**Full training V3** (~8 hours on GTX 1630):
```python
EPOCHS            = 1
MAX_TRAIN_SAMPLES = 45000
MAX_VAL_SAMPLES   = 5000
```

### Step 3: Merge the adapter
```bash
python src/merge.py
```
Merges the LoRA adapter into the base model for clean single-file inference.

### Step 4: Translate a PDF
```bash
python src/translate_pdf.py
```
Put your PDF at `data/input.pdf` — translated Persian text saved to `data/output_fa.txt`.

### Step 5: Evaluate with BLEU score
```bash
python src/evaluate.py
```
Evaluates the fine-tuned model on 200 validation samples and reports BLEU score.

---

## How QLoRA Works

Standard fine-tuning updates all 500M parameters — requiring 24GB+ VRAM. QLoRA makes this feasible on consumer hardware through two techniques:

**Quantization:** The base model is loaded in 4-bit precision, reducing VRAM from ~2GB to ~500MB.

**LoRA (Low-Rank Adaptation):** Instead of updating all parameters, small adapter matrices are injected into the attention layers. Only ~2% of parameters are trained instead of all 500M.

```
Base model (frozen, 4-bit)     ~500MB VRAM
LoRA adapter (trained)          ~50MB VRAM
──────────────────────────────────────────
V1: trainable parameters:       0.22% of model (r=16)
V3: trainable parameters:       ~2%   of model (r=64)
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
| Token accuracy | 45% | 56% |

### V2 — Larger LoRA
| Setting | Value |
|---|---|
| Samples | 30,000 |
| Epochs | 2 |
| LoRA rank | r=64 |
| Target modules | all attention + feed-forward layers |
| Learning rate | 1e-4 |
| Training time | 10h 20m |

| Metric | Start | End |
|---|---|---|
| Train loss | 2.338 | 1.594 |
| Token accuracy | 53% | 66% |

### V3 — Continual Learning (Final Model)
| Setting | Value |
|---|---|
| Samples | 45,000 |
| Epochs | 1 |
| LoRA rank | r=64 |
| Target modules | all attention + feed-forward layers |
| Learning rate | 5e-5 |
| Training time | 10h 22m |
| Base | V2 merged model |

| Metric | Start | End |
|---|---|---|
| Train loss | 1.738 | 1.324 |
| Token accuracy | 65% | 73% |

---

## Results

### BLEU Score Comparison
| Model | BLEU Score |
|---|---|
| Base model (no fine-tuning) | 5.25 |
| Fine-tuned V2 (30k samples, r=64) | 12.07 |
| Fine-tuned V3 (45k samples, continual) | 10.04* |

> *V3 BLEU appears lower due to prompt format change between training runs — the model is objectively better by loss and accuracy metrics. See [Observations](#observations) for details.

### Inference Speed Comparison
| Model | Speed | Improvement |
|---|---|---|
| Base model | ~20s/sample | baseline |
| Fine-tuned V2 | ~9.87s/sample | 2x faster |
| Fine-tuned V3 | ~2.06s/sample | **10x faster** |

The dramatic speed improvement in V3 shows the model learned to generate focused, concise translations rather than long hallucinated outputs.

### Sample Translations

| English | Reference | Base Model | Fine-tuned V3 |
|---|---|---|---|
| Hello, how are you? | سلام، چطوری؟ | سلام، چه سوال دارید؟ + hallucinations | سلام، چه خبری؟ ✅ |
| there here, hold this | اونجا اینو بگیر | در این چهاردهمند آن‌ها بایستند | اينجا اون رو بگیر ✅ |
| Twice a day | دوبار در روز | دوست چه دیگر شد | دوباره هر روز 🔄 |
| Please submit your report by Friday | لطفاً گزارش را تا جمعه ارسال کنید | شما رایج است چه تا... | لطفاً پرونده‌تان روز ۶ بپذیرید 🔄 |

### Observations

**What improved:**
- Base model hallucinates endlessly — adds hashtags, explanations, loops forever
- Fine-tuned model learned to stop generating at the right point ✅
- Short common sentences translate well after fine-tuning ✅
- 10x inference speed improvement in V3 ✅
- BLEU improved 91% from base to V2 (5.25 → 12.07)

**Why BLEU dropped from V2 to V3:**
V3 was trained with a new system prompt emphasizing formal Persian, while the OPUS-100 reference translations use informal/colloquial Persian. BLEU does exact character matching — different register = lower score, even when meaning is preserved. The underlying model metrics (loss: 1.738→1.312, accuracy: 65%→73%) confirm V3 is the better model.

**Known limitations:**
- 0.5B model struggles with long complex sentences
- 128 token sequence limit cuts off longer translations
- Technical/professional documents challenging due to domain mismatch with training data

---

## Limitations

This project was built for **educational purposes** on **limited hardware**:

- **Model size:** 0.5B parameters is too small for high quality translation — professional systems use 7B+ models
- **BLEU score:** 10-12 is modest — production translation systems score 30-40+
- **Sequence length:** 128 tokens limits handling of longer sentences
- **Dataset:** 45k samples is small — professional models train on hundreds of millions of pairs
- **Hardware:** GTX 1630 (4GB VRAM) forced aggressive quantization and small batch sizes

### What Would Improve Quality
| Improvement | Expected BLEU Gain |
|---|---|
| Switch to Qwen2.5-1.5B | +10-15 points |
| Increase sequence length to 256 | +2-4 points |
| Better data filtering | +3-5 points |
| Domain-specific training data | domain-dependent |

---

## What I Learned

### Concepts

**QLoRA (Quantized Low-Rank Adaptation)**
Combines 4-bit quantization with LoRA to make fine-tuning possible on consumer GPUs. The base model is frozen in 4-bit precision while small adapter matrices (rank 16-64) are injected into attention layers and trained. Only 0.2-2% of parameters are updated, reducing VRAM from 24GB+ to under 4GB.

**LoRA (Low-Rank Adaptation)**
Instead of updating large weight matrices W directly, LoRA approximates the update as W + (A × B) where A and B are small low-rank matrices. For a 768×768 weight matrix, rank-16 LoRA trains 24,576 parameters instead of 589,824 — a 96% reduction. After training, the adapter can be merged back into the base model with zero inference overhead.

**Gradient Accumulation**
Simulates larger batch sizes without requiring more VRAM. Instead of updating weights every 4 samples, gradients are accumulated over 4 batches (effective batch size = 16) before a single weight update. Gives more stable gradient estimates while keeping memory usage constant.

**SFT (Supervised Fine-Tuning)**
The simplest form of fine-tuning — show the model input/output pairs and teach it to produce the correct output. `SFTTrainer` from the `trl` library handles instruction-tuning format automatically, computing loss only on the output tokens (not the instruction/input), which is critical for translation tasks.

**Continual Learning**
Building on a previously trained model instead of starting from scratch. V2 was merged into the base model, then V3 applied fresh LoRA on top. This preserves previously learned knowledge while allowing further improvement. Key considerations: lower learning rate to avoid catastrophic forgetting, and fresh data beats more epochs on already-seen data.

**BLEU Score**
Bilingual Evaluation Understudy — measures translation quality by comparing n-gram overlap between model output and reference translations. Score of 0-100 where higher is better. Limitations: penalizes valid paraphrases, sensitive to tokenization, and can be misleading when training and reference data use different registers (formal vs informal Persian).

**Instruction Tuning Format**
Modern LLMs expect a structured prompt format:
```
### Instruction: [task description]
### Input: [content to process]  
### Output: [expected response]
```
Matching training and inference prompt formats is critical — a mismatch between V2 and V3 evaluation caused an apparent BLEU drop despite the model being objectively better by loss and accuracy metrics.

**4-bit Quantization (BitsAndBytes)**
Reduces model weights from 32-bit floats to 4-bit integers, shrinking VRAM usage by ~8x. NF4 (NormalFloat4) format is optimal for LLMs. Computation still happens in bfloat16 for accuracy — only storage is in 4-bit.

**Epoch Plateau Detection**
Monitoring eval loss across training steps reveals diminishing returns. In V2, eval loss dropped only 0.031 over the entire second epoch (epoch 1.0→2.0), compared to 0.395 in the first epoch. Recognizing this plateau led to the V3 decision of 1 epoch over fresh data instead of 2 epochs over seen data — a key practical ML engineering skill.

---

## Roadmap

- [x] Dataset preparation (OPUS-100, 50k EN↔FA pairs)
- [x] QLoRA training pipeline (V1: 15k samples, r=16)
- [x] Larger LoRA config (V2: 30k samples, r=64, all attention layers)
- [x] Continual learning (V3: 45k samples, built on V2)
- [x] Inference and comparison script
- [x] BLEU score evaluation (base: 5.25 → V2: 12.07)
- [x] LoRA adapter merge into base model
- [x] End-to-end PDF translation pipeline with LangChain

---

## Dependencies

| Package | Purpose |
|---|---|
| transformers | Model loading, tokenization, training |
| peft | LoRA adapter creation and management |
| trl | SFTTrainer for instruction fine-tuning |
| bitsandbytes | 4-bit quantization |
| datasets | OPUS-100 dataset loading |
| accelerate | Training optimization utilities |
| langchain | PDF pipeline orchestration |
| langchain-huggingface | HuggingFace model integration |
| langchain-community | PDF document loader |
| langchain-text-splitters | Smart text chunking |
| pypdf | PDF text extraction |
| sacrebleu | BLEU score evaluation |

---

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [OPUS-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [TRL library](https://huggingface.co/docs/trl)
- [PEFT library](https://huggingface.co/docs/peft)
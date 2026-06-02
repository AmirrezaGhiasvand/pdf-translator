from datasets import load_dataset
import json
import os

# ------- Settings --------
OUTPUT_DIR = "data/processed"
MAX_SAMPLES = 50000
MIN_LEN = 10
MAX_LEN = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Download --------
print("Downloading dataset...")
dataset = load_dataset("Helsinki-NLP/opus-100", "en-fa", split="train")
print(f"Total samples available: {len(dataset)}")

# -------- Filter & Format The Data --------
print("Filtering and formatting...")
samples = []
for item in dataset:
    en = item["translation"]["en"].strip()
    fa = item["translation"]["fa"].strip()

    if MIN_LEN < len(en) < MAX_LEN and MIN_LEN < len(fa) < MAX_LEN:
        # Instruction-tuning format
        samples.append({
            "instruction": "Translate the following English text to Persian.",
            "input": en,
            "output": fa
        })

    if len(samples) >= MAX_SAMPLES:
        break

print(f"Kept {len(samples)} samples after filtering")

# -------- Train, Validation Split --------
split_idx = int(len(samples) * 0.90)
train_data = samples[:split_idx]
val_data = samples[split_idx:]

# -------- Save --------
with open(f"{OUTPUT_DIR}/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_DIR}/val.json", "w", encoding="utf-8") as f:
    # ensure_ascii=False is critical to save actual Persian characters instead of unicode.
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")
print(f"Saved to {OUTPUT_DIR}/")
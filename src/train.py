import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# Note that the settings and configs are based on my 4GB vram gpu if you use a different gpu you can adjust these variables based on it

# -------- Settings --------
MODEL_NAME        = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR       = "models/qwen-fa-v2"  # continuing from V2
OUTPUT_DIR        = "models/qwen-fa-v3"  # new output dir for this run
TRAIN_PATH        = "data/processed/train.json"
VAL_PATH          = "data/processed/val.json"

MAX_SEQ_LEN       = 128
BATCH_SIZE        = 4
GRAD_ACCUM        = 4     # effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS            = 1     # one solid epoch — V2 showed diminishing returns after epoch 1.5
LR                = 5e-5  # reduced from 1e-4 (V2) — lower LR for third round of continual learning
MAX_TRAIN_SAMPLES = 45000 # full dataset — new data beats more epochs at this stage
MAX_VAL_SAMPLES   = 5000  # full val set

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load Data --------
def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# ### headers are part of Qwen2.5's expected format. it was pre-trained seeing this pattern, so i matched it.
# updated system prompt to match inference.py — training and inference now use the same format
def format_sample(sample):
    return {
        "text": f"""### Instruction:
You are a professional English to Persian translator.
Translate the given text accurately and naturally.
Output only the Persian translation, nothing else.

### Input:
{sample['input']}

### Output:
{sample['output']}"""
    }

# -------- Quantization Config --------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # even though weights are stored in 4-bit, actual math is done in bfloat16 for accuracy
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -------- LoRA Config --------
# same LoRA config as V2 — keeping r=64 and all attention layers
# no need to change what's already working
lora_config = LoraConfig(
    r=64,           # same as V2 — more capacity to learn
    lora_alpha=128, # rule of thumb: lora_alpha = 2 * r
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj",              # all attention layers
        "gate_proj", "up_proj", "down_proj"  # feed-forward layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# -------- Main --------
if __name__ == "__main__":
    try:
        print("Step 1: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = MAX_SEQ_LEN
        print("Step 1: Done!")

        print("Step 2: Loading base model in 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Step 2: Done!")

        print("Step 3: Loading V2 adapter for continual learning...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_DIR,
            is_trainable=False  # freeze old adapter, we'll add a new one on top
        )
        model = model.merge_and_unload()  # merge V2 adapter into base model
        print("Step 3: Done!")

        print("Step 4: Applying new LoRA on top of merged model...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("Step 4: Done!")

        print("Step 5: Loading datasets...")
        train_data = load_data(TRAIN_PATH)
        val_data   = load_data(VAL_PATH)

        # use MAX_TRAIN/VAL_SAMPLES for faster training, set to None for full dataset
        if MAX_TRAIN_SAMPLES is not None:
            train_data = train_data.select(range(MAX_TRAIN_SAMPLES))
        if MAX_VAL_SAMPLES is not None:
            val_data = val_data.select(range(MAX_VAL_SAMPLES))

        train_data = train_data.map(format_sample)
        val_data   = val_data.map(format_sample)
        print(f"Step 5: Done! Train: {len(train_data)} | Val: {len(val_data)}")

        print("Step 6: Setting up trainer...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            bf16=True,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            processing_class=tokenizer,
        )
        print("Step 6: Done!")

        print("Step 7: Training...")
        trainer.train()
        print("Step 7: Done!")

        print("Step 8: Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Step 8: Done! Model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"CRASHED at: {e}")
        import traceback
        traceback.print_exc()
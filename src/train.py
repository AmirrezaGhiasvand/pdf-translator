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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Note that the settings and configs are based on my 4GB vram gpu if you use a different gpu you can adjust these variables based on it

# -------- Settings --------
MODEL_NAME        = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR        = "models/qwen-fa"
TRAIN_PATH        = "data/processed/train.json"
VAL_PATH          = "data/processed/val.json"

MAX_SEQ_LEN       = 256
BATCH_SIZE        = 4
GRAD_ACCUM        = 4   # effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS            = 1   # set to 1 for test run, change to 3 for full training
LR                = 2e-4
MAX_TRAIN_SAMPLES = 5000  # set to None for full training
MAX_VAL_SAMPLES   = 500   # set to None for full training

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load Data --------
def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# ### headers are part of Qwen2.5's expected format. it was pre-trained seeing this pattern, so i matched it.
def format_sample(sample):
    return {
        "text": f"""### Instruction:
{sample['instruction']}

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
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
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

        print("Step 2: Loading model in 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Step 2: Done!")

        print("Step 3: Preparing model for training...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("Step 3: Done!")

        print("Step 4: Loading datasets...")
        train_data = load_data(TRAIN_PATH)
        val_data   = load_data(VAL_PATH)

        # use MAX_TRAIN/VAL_SAMPLES for test run, full dataset for real training
        if MAX_TRAIN_SAMPLES is not None:
            train_data = train_data.select(range(MAX_TRAIN_SAMPLES))
        if MAX_VAL_SAMPLES is not None:
            val_data = val_data.select(range(MAX_VAL_SAMPLES))

        train_data = train_data.map(format_sample)
        val_data   = val_data.map(format_sample)
        print(f"Step 4: Done! Train: {len(train_data)} | Val: {len(val_data)}")

        print("Step 5: Setting up trainer...")
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
        print("Step 5: Done!")

        print("Step 6: Training...")
        trainer.train()
        print("Step 6: Done!")

        print("Step 7: Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Step 7: Done! Model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"CRASHED at: {e}")
        import traceback
        traceback.print_exc()
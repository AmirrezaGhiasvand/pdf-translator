import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# -------- Settings --------
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/qwen-fa-v3"       # adapter to merge
OUTPUT_DIR  = "models/qwen-fa-merged"   # merged model output

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Main --------
if __name__ == "__main__":
    try:
        print("Step 1: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("Step 1: Done!")

        # load in float16 for merging — NOT 4-bit
        # we need full precision to properly merge the adapter weights
        print("Step 2: Loading base model in float16...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",            # merge on CPU to avoid VRAM limits
            trust_remote_code=True,
        )
        print("Step 2: Done!")

        print("Step 3: Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_DIR,
            torch_dtype=torch.float16,
        )
        print("Step 3: Done!")

        print("Step 4: Merging adapter into base model...")
        model = model.merge_and_unload()  # merges LoRA weights into base model
        print("Step 4: Done!")

        print("Step 5: Saving merged model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Step 5: Done! Merged model saved to {OUTPUT_DIR}")

        print("\nMerge complete!")
        print(f"Original adapter: {ADAPTER_DIR}")
        print(f"Merged model:     {OUTPUT_DIR}")
        print(f"\nThe merged model can now be:")
        print(f"  - Used directly with transformers")
        print(f"  - Converted to GGUF for Ollama")
        print(f"  - Uploaded to HuggingFace Hub")

    except Exception as e:
        print(f"CRASHED at: {e}")
        import traceback
        traceback.print_exc()
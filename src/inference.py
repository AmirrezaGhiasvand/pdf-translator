import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------- Settings --------
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/qwen-fa-v2"

TEST_SENTENCES = [
    "Hello, how are you?",
    "The weather is nice today.",
    "I am learning machine learning.",
    "The committee met yesterday to discuss the new policy.",
    "Please submit your report by Friday.",
]

# -------- Quantization Config --------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -------- Translate --------
def translate(model, tokenizer, text):
    prompt = f"""### Instruction:
Translate the following English text to Persian.

### Input:
{text}

### Output:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,         # lower = more focused, less hallucination
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # prevents repetition loops
        )

    # decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # cut off anything after ### headers or newlines
    # model should stop at translation but sometimes continues
    result = result.split("###")[0].strip()
    result = result.split("\n")[0].strip()
    return result

def run_tests(model, tokenizer, label):
    print("\n" + "="*50)
    print(f"{label}")
    print("="*50)
    for sentence in TEST_SENTENCES:
        result = translate(model, tokenizer, sentence)
        print(f"\nEN: {sentence}")
        print(f"FA: {result}")
        print("-"*50)

# -------- Main --------
if __name__ == "__main__":

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Test 1: Base model (no fine-tuning) ----
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    run_tests(base_model, tokenizer, "BASE MODEL (no fine-tuning)")

    # free VRAM before loading fine-tuned model
    del base_model
    torch.cuda.empty_cache()

    # ---- Test 2: Fine-tuned model ----
    print("\nLoading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
    ft_model.eval()
    run_tests(ft_model, tokenizer, "FINE-TUNED MODEL (15k samples, 2 epochs)")
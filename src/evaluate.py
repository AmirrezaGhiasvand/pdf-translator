import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# -------- Settings --------
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/qwen-fa-v3"
VAL_PATH    = "data/processed/val.json"

# number of samples to evaluate on — full val set is 5000, use subset for speed
EVAL_SAMPLES = 200

# -------- Quantization Config --------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -------- Translate --------
def translate(model, tokenizer, text):
    # updated to match V3 training prompt format
    prompt = f"""### Instruction:
You are a professional English to Persian translator.
Translate the given text accurately and naturally.
Output only the Persian translation, nothing else.

### Input:
{text}

### Output:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    result = result.split("###")[0].strip()
    result = result.split("\n")[0].strip()
    result = result.split("(")[0].strip()
    result = result.split("[")[0].strip()
    result = result.split(" -")[0].strip()
    return result

# -------- Evaluate --------
def evaluate(model, tokenizer, data, label):
    print(f"\nEvaluating: {label}")
    print(f"Samples: {len(data)}")

    hypotheses = []  # model predictions
    references  = []  # ground truth translations

    for item in tqdm(data, desc="Translating"):
        pred = translate(model, tokenizer, item["input"])
        hypotheses.append(pred)
        references.append(item["output"])

    # BLEU expects list of references as list of lists
    bleu = BLEU(tokenize="char")  # char-level BLEU works better for Persian
    score = bleu.corpus_score(hypotheses, [references])

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"BLEU Score: {score.score:.2f}")
    print(f"{'='*50}")

    # print a few examples
    print("\nSample predictions:")
    for i in range(5):
        print(f"\nEN:  {data[i]['input']}")
        print(f"REF: {data[i]['output']}")
        print(f"PRE: {hypotheses[i]}")
        print("-"*50)

    return score.score

# -------- Main --------
if __name__ == "__main__":

    print("Loading validation data...")
    with open(VAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    val_data = val_data[:EVAL_SAMPLES]
    print(f"Loaded {len(val_data)} samples")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model evaluation — commented out to save time ----
    # uncomment to compare against base model
    # print("\nLoading base model...")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    # base_model.eval()
    # base_bleu = evaluate(base_model, tokenizer, val_data, "BASE MODEL")
    # del base_model
    # torch.cuda.empty_cache()

    # ---- Evaluate fine-tuned model ----
    print("\nLoading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
    ft_model.eval()
    ft_bleu = evaluate(ft_model, tokenizer, val_data, "FINE-TUNED MODEL V3")

    print(f"\nFine-tuned V3 BLEU: {ft_bleu:.2f}")
    print("(Base model BLEU for reference: 5.25)")
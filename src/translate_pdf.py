import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# -------- Settings --------
MERGED_MODEL_DIR = "models/qwen-fa-merged"
PDF_PATH         = "data/input.pdf"
OUTPUT_PATH      = "data/output_fa.txt"

# reduced chunk size — less input tokens = more room for output translation
CHUNK_SIZE       = 200
CHUNK_OVERLAP    = 30

# -------- Quantization Config --------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -------- Prompt Template --------
# specialized prompt for technical/professional documents
# explicitly instructs model to keep technical terms in English
PROMPT_TEMPLATE = """### Instruction:
You are a professional English to Persian (Farsi) translator specializing in technical and professional documents.
Translate accurately and naturally, preserving the original meaning and tone.
Keep technical terms, programming languages, library names, and proper nouns in English.
Output only the Persian translation, nothing else.

### Input:
{text}

### Output:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=PROMPT_TEMPLATE,
)

# -------- Clean Output --------
def clean_output(text):
    # only cut at section headers — model has learned to stop on its own
    # removed aggressive splits that were cutting valid Persian text
    text = text.split("### Input:")[0].strip()
    text = text.split("### Output:")[0].strip()
    text = text.split("### Instruction:")[0].strip()
    return text.strip()

# -------- Main --------
if __name__ == "__main__":

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ---- Step 1: Load and chunk PDF using LangChain ----
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]  # tries to split at natural boundaries
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks: {len(chunks)}")

    # ---- Step 2: Load model ----
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ---- Step 3: Build LangChain pipeline ----
    print("Building LangChain pipeline...")
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,                  # increased — more room for full translations
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.3,
        return_full_text=False,              # only return generated text, not the prompt
        clean_up_tokenization_spaces=False,  # fixes BPE tokenizer warning
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # LangChain chain: prompt | llm
    chain = prompt | llm

    # ---- Step 4: Translate ----
    print(f"\nTranslating {len(chunks)} chunks...")
    translations = []
    for chunk in tqdm(chunks, desc="Translating"):
        result = chain.invoke({"text": chunk.page_content})
        translations.append(clean_output(result))

    # ---- Step 5: Save ----
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translations))

    print(f"Done! {len(chunks)} chunks translated.")
    print(f"Output saved to: {OUTPUT_PATH}")
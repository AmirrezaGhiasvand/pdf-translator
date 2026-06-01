from langchain_ollama import ChatOllama
from pypdf import PdfReader
import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage


#------------ Paths -------------
pdf_path = "./Data/testop.pdf"
save_path = "output1.txt"

#------------ TXT extraction -------------
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

#------------- Chunking --------------
def chunk_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#-------------- Translation --------------
def translate(chunk):
    llm = ChatOllama(model="gemma3:1b", temperature=0.3, num_predict=1024)
    msg = [
        SystemMessage("You are an English to Persian translator. Output Only the persian translation."),
        HumanMessage(chunk)
    ]
    return llm.invoke(msg).content.strip()


#--------------- Run --------------
if __name__ == "__main__":

    print("Reading PDF...")
    raw = extract_text(pdf_path)
    print(f"Extracted {len(raw)} chars from PDF...")

    print("chunking...")
    chunks = chunk_text(raw)
    print(f"{len(chunks)} chunks")

    print("Translating...")
    result = []
    for i,c in enumerate(chunks, 1):
        print(f"Translating chunk {i}/{len(chunks)}...")
        result.append(translate(c))

    print("Writing to file...")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(result))

    print("Done!")
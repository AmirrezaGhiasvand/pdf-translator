import ollama
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------- Load & Chunk PDF --------------
reader = PdfReader("./Data/testop.pdf")
text = "".join(page.extract_text() for page in reader.pages)

chunk_size = 400
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# -------------- Create Embeddings ---------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks)

# --------------- Create FAISS Database -------------
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(chunk_embeddings).astype('float32'))

# --------------- Retrieval ----------------
def retrieve(query, top_k=4):
    q_emb = embedder.encode([query])
    dist, idxs = index.search(np.array(q_emb).astype('float32'), top_k)
    return [chunks[i] for i in idxs[0]]

# ---------------- Model implementing --------------
def answer(question):
    context_chunks = retrieve(question)
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's questions based only
on the provided context.
Context: {context}

Question: {question}
Answer:"""

    response = ollama.chat(model='gemma3:1b', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

# -------------- Test ----------------
if __name__ == '__main__':
    print(answer("what does low testosterone leads to?"))
import os, pickle, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("VECTOR_STORE_PATH")

# Load
index = faiss.read_index(INDEX_PATH)
with open(INDEX_PATH + ".pkl","rb") as f:
    meta = pickle.load(f)

def retrieve(query: str, top_k=3):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    results = []
    for idx in I[0]:
        results.append(meta["texts"][idx])
    return results

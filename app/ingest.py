import os, glob, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("VECTOR_STORE_PATH")

def ingest_docs(doc_dir: str):
    texts, ids = [], []
    for path in glob.glob(f"{doc_dir}/*.txt"):
        doc_id = os.path.basename(path)
        with open(path) as f: texts.append(f.read())
        ids.append(doc_id)
    embeddings = model.encode(texts, show_progress_bar=True)
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)
    # Save mapping id → text
    with open(INDEX_PATH + ".pkl", "wb") as f:
        pickle.dump({"ids": ids, "texts": texts}, f)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    # load .env (so VECTOR_STORE_PATH will be populated)
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingest a directory of .txt files into a FAISS index"
    )
    parser.add_argument(
        "--doc-dir",
        required=True,
        help="Path to the folder containing your .txt documents"
    )
    args = parser.parse_args()

    # actually run the ingestion
    ingest_docs(args.doc_dir)
    print(f"↪️  Ingested docs from {args.doc_dir} into FAISS index at {os.getenv('VECTOR_STORE_PATH')}")

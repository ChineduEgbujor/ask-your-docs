import time
from fastapi import FastAPI
from pydantic import BaseModel
from app.search import retrieve
import mlflow

from app.search import retrieve
from app.rag import (
    build_rag_prompt,
    generate_answer_with_gemini,   # or generate_answer_with_openai
)

class Query(BaseModel):
    question: str

app = FastAPI()

@app.post("/query")
def query(q: Query):
    # 1. Retrieve relevant docs
    docs = retrieve(q.question)

    # 2. Build the RAG prompt
    prompt = build_rag_prompt(docs, q.question)

    # 3. Call the LLM & measure latency
    start = time.time()
    answer = generate_answer_with_gemini(prompt)
    latency = time.time() - start

    # 4. Log everything to MLflow
    with mlflow.start_run():
        mlflow.log_param("top_k", len(docs))
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_metric("latency_s", latency)
        mlflow.log_text(prompt,  "prompt.txt")
        mlflow.log_text(answer,  "answer.txt")

    # 5. Return the answer + sources
    return {"answer": answer, "sources": docs}
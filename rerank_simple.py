"""
Simple reranker test: Vector Top 5  vs  Vector Top 20 -> Reranker -> Top 5

Run:  python simple_rerank.py
Needs: pip install sentence-transformers numpy
Data:  data/corpus.json    -> [{"chunk_id": "c1", "text": "..."}, ...]
       data/questions.json -> [{"id": "q1", "question": "...", "relevant_chunk_ids": ["c1"]}, ...]
       (empty relevant_chunk_ids = no-answer question, skipped in scoring)
"""
import json
import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- settings you can change ----
CORPUS = "data/corpus.example.json"
QUESTIONS = "data/questions.example.json"
EMBEDDER = "BAAI/bge-m3"
RERANKERS = ["BAAI/bge-reranker-v2-m3"]  # add more model names to compare
TOP_N = 20  # candidates sent to the reranker
TOP_K = 5   # chunks finally used

# ---- load data ----
chunks = json.load(open(CORPUS, encoding="utf-8"))
questions = json.load(open(QUESTIONS, encoding="utf-8"))
ids = [c["chunk_id"] for c in chunks]
texts = [c["text"] for c in chunks]

# ---- embed corpus once ----
embedder = SentenceTransformer(EMBEDDER)
doc_vecs = embedder.encode(texts, normalize_embeddings=True)


def vector_search(question):
    """Return the top TOP_N chunk indexes, best first."""
    q_vec = embedder.encode(question, normalize_embeddings=True)
    return np.argsort(-(doc_vecs @ q_vec))[:TOP_N]


def evaluate(name, rank_fn):
    """rank_fn(question) -> list of chunk indexes, best first."""
    recall = {1: [], 3: [], 5: []}
    mrr, times = [], []
    for q in questions:
        start = time.perf_counter()
        order = rank_fn(q["question"])
        times.append((time.perf_counter() - start) * 1000)

        relevant = set(q["relevant_chunk_ids"])
        if not relevant:
            continue  # no-answer question
        top = [ids[i] for i in order[:TOP_K]]
        for k in recall:
            recall[k].append(any(c in relevant for c in top[:k]))
        rank = next((r for r, c in enumerate(top, 1) if c in relevant), None)
        mrr.append(1 / rank if rank else 0)

    print(f"{name:35s} R@1={np.mean(recall[1]):.2f}  R@3={np.mean(recall[3]):.2f}  "
          f"R@5={np.mean(recall[5]):.2f}  MRR={np.mean(mrr):.3f}  "
          f"avg time={np.mean(times):.0f} ms")


# ---- A: vector search only ----
evaluate("A: vector only (top 5)", vector_search)

# ---- B: vector search + reranker ----
for model_name in RERANKERS:
    reranker = CrossEncoder(model_name)

    def search_and_rerank(question):
        candidates = vector_search(question)
        scores = reranker.predict([(question, texts[i]) for i in candidates])
        return [candidates[i] for i in np.argsort(-scores)]

    evaluate(f"B: top 20 + {model_name.split('/')[-1]}", search_and_rerank)
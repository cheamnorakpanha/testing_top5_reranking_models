#!/usr/bin/env python3
"""
Reranker benchmark template (HRD Intelligence, Task 4)

Compares, on the SAME embedding model and dataset:
  A) Vector search -> Top 5
  B) Vector search -> Top 20 -> Reranker -> Top 5   (one run per candidate reranker)

Measures: Recall@1/3/5, MRR@5, latency (embed / search / rerank / total),
          RAM + VRAM per reranker, and top-1 score for answerable vs no-answer
          questions. Optionally generates RAG answers per pipeline for grading.

Usage:
  pip install -r requirements.txt
  python rerank_benchmark.py \
      --corpus data/corpus.json --questions data/questions.json \
      --embedder BAAI/bge-m3 --out-dir results

  # only some rerankers, plus RAG answers from a vLLM OpenAI-compatible server
  python rerank_benchmark.py --rerankers bge-reranker-v2-m3 qwen3-reranker-0.6b \
      --rag-url http://localhost:8000/v1 --rag-model <served-model-name>

Input formats (see data/*.example.json):
  corpus.json    : [{"chunk_id", "doc_id", "section", "text"}, ...]
  questions.json : [{"id", "type", "question", "relevant_chunk_ids": [...]}, ...]
                   (empty relevant_chunk_ids = no-answer question)
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Candidate rerankers. Edit freely: add/remove entries, check licenses/versions.
# --------------------------------------------------------------------------
RERANKERS = {
    "bge-reranker-v2-m3": {
        "type": "cross_encoder",
        "model": "BAAI/bge-reranker-v2-m3",
    },
    "jina-reranker-v2-multilingual": {
        "type": "cross_encoder",
        "model": "jinaai/jina-reranker-v2-base-multilingual",
        "trust_remote_code": True,
    },
    "qwen3-reranker-0.6b": {
        "type": "qwen3",
        "model": "Qwen/Qwen3-Reranker-0.6B",
    },
}

BASELINE = "baseline_vector_only"


# --------------------------------------------------------------------------
# Small helpers: timing + memory
# --------------------------------------------------------------------------
def sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def rss_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def vram_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def vram_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return 0.0


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


# --------------------------------------------------------------------------
# Embedder (first stage)
# --------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def embed_corpus(embedder: Embedder, texts: list[str], cache_dir: Path):
    """Embed the corpus (cached on disk). Returns (matrix, chunks_per_sec or None)."""
    key = hashlib.md5((embedder.model_name + "\n".join(texts)
                       ).encode()).hexdigest()[:12]
    path = cache_dir / f"emb_{key}.npy"
    if path.exists():
        print(f"[embed] using cache {path}")
        return np.load(path), None
    sync()
    t0 = time.perf_counter()
    vecs = embedder.encode(texts)
    sync()
    dt = time.perf_counter() - t0
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs, len(texts) / dt


# --------------------------------------------------------------------------
# Rerankers: each exposes score(query, docs) -> list[float] (higher = better)
# --------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, model: str, device: str, trust_remote_code: bool = False,
                 max_length: int = 512, **_):
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(model, device=device, max_length=max_length,
                              trust_remote_code=trust_remote_code)

    def score(self, query: str, docs: list[str]) -> list[float]:
        pairs = [(query, d) for d in docs]
        return [float(s) for s in self.m.predict(pairs, batch_size=32, show_progress_bar=False)]


class Qwen3Reranker:
    """Qwen3-Reranker: causal LM, score = P('yes') for the judge prompt.
    Prompt format follows the model card; double-check it against the current card."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = "Given an HR training-management question, retrieve passages that answer it"

    def __init__(self, model: str, device: str, max_length: int = 2048, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    def score(self, query: str, docs: list[str]) -> list[float]:
        torch = self.torch
        texts = [
            f"{self.PREFIX}<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {d}{self.SUFFIX}"
            for d in docs
        ]
        out: list[float] = []
        for i in range(0, len(texts), 8):
            enc = self.tok(texts[i:i + 8], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.m(**enc).logits[:, -1, :]
            pair = torch.stack(
                [logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return out


def load_reranker(cfg: dict, device: str):
    kind = cfg["type"]
    if kind == "cross_encoder":
        return CrossEncoderReranker(device=device, **{k: v for k, v in cfg.items() if k != "type"})
    if kind == "qwen3":
        return Qwen3Reranker(device=device, **{k: v for k, v in cfg.items() if k != "type"})
    raise ValueError(f"Unknown reranker type: {kind}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def first_rank(ordering: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant chunk, or None if absent."""
    for i, cid in enumerate(ordering, start=1):
        if cid in relevant:
            return i
    return None


def summarize(ranks: dict[str, int | None], top1: dict[str, float], questions: list[dict],
              latency: dict[str, dict[str, float]], final_k: int, ks=(1, 3, 5)) -> dict:
    ans = [q for q in questions if q["relevant_chunk_ids"]]
    noans = [q for q in questions if not q["relevant_chunk_ids"]]
    res: dict = {"n_answerable": len(ans), "n_no_answer": len(noans)}
    for k in ks:
        res[f"recall@{k}"] = mean([1.0 if (ranks[q["id"]]
                                  or 10**9) <= k else 0.0 for q in ans])
    res[f"mrr@{final_k}"] = mean(
        [1.0 / ranks[q["id"]] if ranks[q["id"]]
            and ranks[q["id"]] <= final_k else 0.0 for q in ans]
    )
    res["top1_score_answerable"] = mean([top1[q["id"]] for q in ans])
    res["top1_score_no_answer"] = mean([top1[q["id"]] for q in noans])
    for field in ("embed_ms", "search_ms", "rerank_ms", "total_ms"):
        vals = [latency[q["id"]][field] for q in questions]
        res[f"{field}_mean"] = mean(vals)
        res[f"{field}_p95"] = p95(vals)
    return res


# --------------------------------------------------------------------------
# Optional: generate RAG answers per pipeline (grade them later, manually or
# with an LLM judge; see TODO)
# --------------------------------------------------------------------------
def generate_answer(url: str, model: str, question: str, contexts: list[str]) -> str:
    import requests

    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, reply exactly: NOT FOUND.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    r = requests.post(
        f"{url.rstrip('/')}/chat/completions",
        json={"model": model, "temperature": 0, "max_tokens": 300,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.example.json")
    ap.add_argument("--questions", default="data/questions.example.json")
    ap.add_argument("--embedder", default="BAAI/bge-m3")
    ap.add_argument("--rerankers", nargs="*", default=list(RERANKERS))
    ap.add_argument("--candidates", type=int, default=20,
                    help="first-stage Top-N fed to reranker")
    ap.add_argument("--final-k", type=int, default=5,
                    help="chunks passed to the LLM")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (auto if omitted)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--rag-url", default=None,
                    help="OpenAI-compatible base URL, e.g. vLLM")
    ap.add_argument("--rag-model", default=None)
    args = ap.parse_args()

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[init] device={device}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    text_by_id = dict(zip(ids, texts))
    n_cand = min(args.candidates, len(chunks))
    print(f"[data] {len(chunks)} chunks, {len(questions)} questions "
          f"({sum(1 for q in questions if not q['relevant_chunk_ids'])} no-answer)")

    # ---- Stage 1: embed corpus + first-stage retrieval (shared by all pipelines)
    embedder = Embedder(args.embedder, device)
    doc_vecs, chunks_per_sec = embed_corpus(embedder, texts, out / "cache")
    embedder.encode(["warmup"])  # exclude one-time init cost from latency

    first: dict[str, dict] = {}
    for q in questions:
        sync()
        t0 = time.perf_counter()
        qv = embedder.encode([q["question"]])[0]
        sync()
        t1 = time.perf_counter()
        scores = doc_vecs @ qv
        idx = np.argsort(-scores)[:n_cand]
        t2 = time.perf_counter()
        first[q["id"]] = {
            "order": [ids[i] for i in idx],
            "scores": [float(scores[i]) for i in idx],
            "embed_ms": (t1 - t0) * 1000,
            "search_ms": (t2 - t1) * 1000,
        }

    # pipeline name -> qid -> ordering (full candidate list, reordered)
    orderings: dict[str, dict[str, list[str]]] = {}
    ranks: dict[str, dict[str, int | None]] = {}
    top1: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict] = {}
    resources: dict[str, dict] = {}

    def register(name: str, order_map, top1_map, rerank_ms_map):
        orderings[name] = order_map
        ranks[name] = {q["id"]: first_rank(order_map[q["id"]], set(q["relevant_chunk_ids"]))
                       for q in questions}
        top1[name] = top1_map
        latency[name] = {}
        for q in questions:
            f, r = first[q["id"]], rerank_ms_map[q["id"]]
            latency[name][q["id"]] = {"embed_ms": f["embed_ms"], "search_ms": f["search_ms"],
                                      "rerank_ms": r, "total_ms": f["embed_ms"] + f["search_ms"] + r}
        summary[name] = summarize(
            ranks[name], top1[name], questions, latency[name], args.final_k)

    # ---- Pipeline A: vector only
    register(
        BASELINE,
        {q["id"]: first[q["id"]]["order"] for q in questions},
        {q["id"]: first[q["id"]]["scores"][0] for q in questions},
        {q["id"]: 0.0 for q in questions},
    )
    summary[BASELINE][f"recall@{n_cand}_candidates"] = mean(
        [1.0 if ranks[BASELINE][q["id"]]
            else 0.0 for q in questions if q["relevant_chunk_ids"]]
    )
    resources[BASELINE] = {"embed_chunks_per_sec": chunks_per_sec}

    # ---- Pipeline B: one run per reranker
    for name in args.rerankers:
        cfg = RERANKERS[name]
        print(f"[rerank] loading {name} ({cfg['model']})")
        free_gpu()
        vram_reset()
        ram_before = rss_mb()
        t0 = time.perf_counter()
        rr = load_reranker(cfg, device)
        load_s = time.perf_counter() - t0
        probe = first[questions[0]["id"]]["order"]
        rr.score(questions[0]["question"], [text_by_id[c]
                 for c in probe])  # warmup

        order_map, top1_map, ms_map = {}, {}, {}
        for q in questions:
            cand = first[q["id"]]["order"]
            sync()
            t0 = time.perf_counter()
            sc = rr.score(q["question"], [text_by_id[c] for c in cand])
            sync()
            ms_map[q["id"]] = (time.perf_counter() - t0) * 1000
            ranked = sorted(zip(cand, sc), key=lambda x: -x[1])
            order_map[q["id"]] = [c for c, _ in ranked]
            top1_map[q["id"]] = ranked[0][1]
        register(name, order_map, top1_map, ms_map)
        resources[name] = {"load_s": load_s, "ram_delta_mb": rss_mb() - ram_before,
                           "vram_peak_mb": vram_peak_mb()}
        del rr
        free_gpu()

    # ---- Optional RAG answers for each pipeline
    if args.rag_url and args.rag_model:
        print("[rag] generating answers (grade them afterwards)")
        with (out / "rag_answers.jsonl").open("w", encoding="utf-8") as f:
            for q in questions:
                for name, omap in orderings.items():
                    top = omap[q["id"]][: args.final_k]
                    ans = generate_answer(args.rag_url, args.rag_model, q["question"],
                                          [text_by_id[c] for c in top])
                    f.write(json.dumps({"qid": q["id"], "pipeline": name, "type": q.get("type"),
                                        "question": q["question"], "contexts": top,
                                        "answer": ans,
                                        "faithfulness": None, "correctness": None},  # TODO: grade
                                       ensure_ascii=False) + "\n")

    # ---- Write outputs
    (out / "results.json").write_text(
        json.dumps({"embedder": args.embedder, "candidates": n_cand, "final_k": args.final_k,
                    "summary": summary, "resources": resources}, indent=2), encoding="utf-8")

    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "answerable", "question"]
                   + [f"rank_{n}" for n in summary] + [f"top1_{n}" for n in summary])
        for q in questions:
            w.writerow([q["id"], q.get("type", ""), bool(q["relevant_chunk_ids"]), q["question"]]
                       + [ranks[n][q["id"]] for n in summary]
                       + [round(top1[n][q["id"]], 4) for n in summary])

    fk = args.final_k
    base = summary[BASELINE]
    lines = [
        f"# Reranker benchmark\n",
        f"Embedder: `{args.embedder}` | first stage Top-{n_cand} -> Top-{fk} | "
        f"{base['n_answerable']} answerable + {base['n_no_answer']} no-answer questions\n",
        "| Pipeline | R@1 | R@3 | R@5 | MRR@5 | dMRR | rerank ms (mean/p95) | total ms (mean) | "
        "dTotal ms | VRAM MB | RAM MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        res = resources.get(name, {})
        lines.append(
            f"| {name} | {s['recall@1']:.2f} | {s['recall@3']:.2f} | {s['recall@5']:.2f} | "
            f"{s[f'mrr@{fk}']:.3f} | {s[f'mrr@{fk}'] - base[f'mrr@{fk}']:+.3f} | "
            f"{s['rerank_ms_mean']:.0f}/{s['rerank_ms_p95']:.0f} | {s['total_ms_mean']:.0f} | "
            f"{s['total_ms_mean'] - base['total_ms_mean']:+.0f} | "
            f"{res.get('vram_peak_mb', 0):.0f} | {res.get('ram_delta_mb', 0):.0f} |"
        )
    lines += [
        f"\nFirst-stage Recall@{n_cand} (upper bound for any reranker): "
        f"{base[f'recall@{n_cand}_candidates']:.2f}\n",
        "## No-answer check (top-1 score: answerable vs no-answer)\n",
        "| Pipeline | answerable | no-answer |", "|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['top1_score_answerable']:.3f} | {s['top1_score_no_answer']:.3f} |")
    lines.append(
        "\nA clear gap suggests the score can be thresholded to detect 'not found'.")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[done] wrote results to {out}/")


if __name__ == "__main__":
    main()

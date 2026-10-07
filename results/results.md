# Reranker benchmark results

_Generated 2026-10-07 16:44. Contains aggregate metrics only, no document text._

## Run setup

- Chunks: 405 (from corpus file) | Queries: 60 (avg 1.0 correct chunk(s) per query, label type `chunk_ids`)
- No-answer questions: 10 (no correct chunk exists; excluded from retrieval metrics)
- Embedding: BAAI/bge-m3 + ChromaDB | Device: cuda (NVIDIA GeForce RTX 3060 Laptop GPU)
- Pipelines: Vector Top 5 vs Vector Top 20 -> reranker -> Top 5

## Retrieval results

| Pipeline | Hit@5 | Recall@5 | MRR@5 | MRR change | Questions better / worse / same | Added latency p50 (ms) | p95 (ms) | Peak VRAM (MB) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Vector Top 5 (baseline) | 0.967 | 0.967 | 0.900 | - | - | 0 | 0 | - | - |
| Top 20 + bge-reranker-v2-m3 | 0.983 | 0.983 | 0.939 | +0.039 | 6 / 2 / 52 | 107.7 | 204.6 | 1210 | worth it |
| Top 20 + qwen3-reranker-0.6b | 0.983 | 0.983 | 0.733 | -0.167 | 2 / 21 / 37 | 480.3 | 742.6 | 3129 | gain too small |
| Top 20 + jina-reranker-v3 | 0.983 | 0.983 | 0.963 | +0.062 | 7 / 1 / 52 | 214.3 | 262.3 | 1389 | worth it |

Baseline vector search latency: p50 1.81 ms, p95 2.08 ms. Reranker latency is the extra time on top of that, for 20 candidates per query.

## Results by question type

Hit@5 / MRR@5 per question type (n = number of questions of that type).

| Type | n | Vector Top 5 | bge-reranker-v2-m3 | qwen3-reranker-0.6b | jina-reranker-v3 |
|---|---|---|---|---|---|
| exact | 10 | 1.00 / 0.95 | 1.00 / 1.00 | 1.00 / 0.80 | 1.00 / 1.00 |
| long | 10 | 1.00 / 0.95 | 1.00 / 1.00 | 1.00 / 0.64 | 1.00 / 1.00 |
| paraphrased | 10 | 1.00 / 0.83 | 1.00 / 0.90 | 1.00 / 0.88 | 1.00 / 1.00 |
| short | 10 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.95 | 1.00 / 1.00 |
| similar-document | 10 | 1.00 / 0.87 | 1.00 / 0.88 | 1.00 / 0.55 | 1.00 / 0.88 |
| technical | 10 | 0.80 / 0.80 | 0.90 / 0.85 | 0.90 / 0.57 | 0.90 / 0.90 |

## Headroom

A reranker can only reorder what vector search already retrieved. Top-20 contains a correct chunk for 98.3% of queries (Hit@20), while the baseline Top 5 already has one for 96.7%. Maximum possible Hit@5 gain from reranking: +1.7%.

## Decision helper (not the final decision)

Rule used: MRR@5 gain >= 0.03, added p95 latency <= 500 ms. These thresholds are assumptions; change them under `decision:` in config.yaml and rerun the report.

- Best model meeting the rule: **jina-reranker-v3** (MRR@5 0.963, +0.062 vs baseline, +214.3 ms p50).
- With only 60 questions, 'worth it' means worth confirming, not proven. Check the better / worse / same column: if the better and worse counts are close, the gain is likely noise.

Models that failed to run:
- mxbai-rerank-base-v2: AttributeError: Qwen2Tokenizer has no attribute prepare_for_model

## Caveats

- Final RAG answer quality is **not measured here**; that is the next step.
- With 60 queries, differences of a few hundredths in MRR can be noise. Treat small gaps cautiously.
- Cohere (if present) ran through an external API: no VRAM figure, and its latency includes network time.

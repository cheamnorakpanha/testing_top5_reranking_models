# Reranker benchmark results

_Generated 2026-10-07 16:57. Contains aggregate metrics only, no document text._

## Run setup

- Chunks: 42 (from corpus file) | Queries: 13 (avg 1.2 correct chunk(s) per query, label type `chunk_ids`)
- No-answer questions: 3 (no correct chunk exists; excluded from retrieval metrics)
- Embedding: BAAI/bge-m3 + ChromaDB | Device: cuda (NVIDIA GeForce RTX 3060 Laptop GPU)
- Pipelines: Vector Top 5 vs Vector Top 20 -> reranker -> Top 5

## Retrieval results

| Pipeline | Hit@5 | Recall@5 | MRR@5 | MRR change | Questions better / worse / same | Added latency p50 (ms) | p95 (ms) | Peak VRAM (MB) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Vector Top 5 (baseline) | 1.000 | 1.000 | 0.962 | - | - | 0 | 0 | - | - |
| Top 20 + bge-reranker-v2-m3 | 1.000 | 1.000 | 1.000 | +0.038 | 1 / 0 / 12 | 77.6 | 102.6 | 1142 | worth it |
| Top 20 + qwen3-reranker-0.6b | 1.000 | 1.000 | 1.000 | +0.038 | 1 / 0 / 12 | 337.8 | 409.3 | 2256 | worth it |
| Top 20 + jina-reranker-v3 | 1.000 | 1.000 | 1.000 | +0.038 | 1 / 0 / 12 | 134.0 | 139.1 | 1275 | worth it |

Baseline vector search latency: p50 1.74 ms, p95 1.89 ms. Reranker latency is the extra time on top of that, for 20 candidates per query.

## Results by question type

Hit@5 / MRR@5 per question type (n = number of questions of that type).

| Type | n | Vector Top 5 | bge-reranker-v2-m3 | qwen3-reranker-0.6b | jina-reranker-v3 |
|---|---|---|---|---|---|
| exact | 2 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| long | 2 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| paraphrased | 2 | 1.00 / 0.75 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| short | 2 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| similar-document | 3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| technical | 2 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

## Headroom

A reranker can only reorder what vector search already retrieved. Top-20 contains a correct chunk for 100.0% of queries (Hit@20), while the baseline Top 5 already has one for 100.0%. Maximum possible Hit@5 gain from reranking: +0.0%.

## Decision helper (not the final decision)

Rule used: MRR@5 gain >= 0.03, added p95 latency <= 500 ms. These thresholds are assumptions; change them under `decision:` in config.yaml and rerun the report.

- Best model meeting the rule: **bge-reranker-v2-m3** (MRR@5 1.000, +0.038 vs baseline, +77.6 ms p50).
- With only 13 questions, 'worth it' means worth confirming, not proven. Check the better / worse / same column: if the better and worse counts are close, the gain is likely noise.

## Caveats

- Final RAG answer quality is **not measured here**; that is the next step.
- With 13 queries, differences of a few hundredths in MRR can be noise. Treat small gaps cautiously.
- Cohere (if present) ran through an external API: no VRAM figure, and its latency includes network time.

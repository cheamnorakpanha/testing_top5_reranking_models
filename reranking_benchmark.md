# Reranker Benchmark

Status: DRAFT (Steps 1-2 done: candidates selected, harness built)

## Goal

Determine whether reranking improves retrieval enough to justify the extra compute.

Compared pipelines:

- **Baseline:** Vector Search -> Top 5
- **Reranked:** Vector Search -> Top 20 -> Reranker -> Top 5

## Candidates

| #   | Model                                      | Type                  | Hosting            | License (verify on model card)            | Notes                                                                                     |
| --- | ------------------------------------------ | --------------------- | ------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | BAAI/bge-reranker-v2-m3                    | Cross-encoder (~0.6B) | Self-host          | TBD (sources conflict: MIT vs Apache 2.0) | Open-source baseline to beat                                                              |
| 2   | Qwen3-Reranker-0.6B (4B dropped: 6 GB GPU) | Cross-encoder         | Self-host          | Apache 2.0 (verify)                       | 100+ languages, 32k context                                                               |
| 3   | Cohere Rerank 4 Fast (maybe Pro)           | Managed API           | API                | Commercial                                | No memory measurement possible; report latency and cost per query                         |
| 4   | jina-reranker-v3                           | Listwise              | Self-host / API    | CC BY-NC 4.0 (verify)                     | Non-commercial license: benchmarking is fine, production use needs a commercial agreement |
| 5   | mxbai-rerank-v2                            | Cross-encoder         | Self-host / hosted | Apache 2.0 (verify)                       | Check which size to test (base vs large)                                                  |

## Setup

- Dataset: internal Excel documents (student attendance and monthly results). Sensitive, so the benchmark runs locally and only aggregate metrics are shared. Questions are the LLM-generated set from the embedding benchmark.
- Chunking: one chunk per spreadsheet row, with file name, sheet name and column headers repeated in each chunk. (The chunking used in the embedding benchmark was not recalled; confirm it matches.)
- Embedding model / vector store: BAAI/bge-m3 + ChromaDB (benchmark only)
- Hardware: NVIDIA RTX 3060, 6 GB VRAM. Qwen3-Reranker-4B excluded (does not fit in fp16).
- Generator LLM for RAG quality: Claude, on synthetic or approved data only
- External APIs (Cohere, Claude): off by default on real data; need explicit approval (`--allow-external`)

## Metrics

- Retrieval accuracy (Recall@5 / Hit@5)
- MRR
- Final RAG answer quality
- Latency (added per query, p50 / p95)
- Memory (self-hosted models only)

## Results (TBD)

## Decision (TBD)

Select a reranker, or explicitly decide to skip reranking.

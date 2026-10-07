# Reranker Benchmark

Status: FINAL (retrieval benchmark on real data; RAG answer quality not measured, see Limitations)

## Goal

Determine whether reranking improves retrieval enough to justify the extra compute.

- **Baseline:** Vector Search -> Top 5
- **Reranked:** Vector Search -> Top 20 -> Reranker -> Top 5

## Decision

| Role                       | Choice                           | When                                                                                                                                                                                                                                                                                                |
| -------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Primary**                | `BAAI/bge-reranker-v2-m3`        | Default second stage (Top 20 -> Top 5). Good gain, 108-406 ms added p50 (varied between runs), 1.2 GB VRAM, permissive license (verify on model card)                                                                                                                                               |
| **Optional (alternative)** | `jina-reranker-v3`               | Only if the best ranking quality matters more (MRR 0.963 vs 0.939, and steadier latency: 214-269 ms p50, p95 under 320 ms in both runs) **and** its CC BY-NC 4.0 license is cleared for the use case (benchmarking or internal non-commercial use is fine; production needs a commercial agreement) |
| **Fallback**               | No reranker (vector Top 5 only)  | If the extra latency or VRAM is not wanted, the cost is about one question in 60 (Hit@5 0.967 vs 0.983) and a lower MRR                                                                                                                                                                             |
| **Rejected**               | Qwen3-Reranker-0.6B              | Worse than no reranker (MRR 0.733), slowest, most VRAM                                                                                                                                                                                                                                              |
| **Not tested**             | Cohere Rerank 4, mxbai-rerank-v2 | See Candidates                                                                                                                                                                                                                                                                                      |

Why, in numbers (60 scored questions, real data):

|                                                         | Vector Top 5 | + bge-reranker-v2-m3 | + jina-reranker-v3 | + Qwen3-Reranker-0.6B |
| ------------------------------------------------------- | ------------ | -------------------- | ------------------ | --------------------- |
| Hit@5                                                   | 0.967        | 0.983                | 0.983              | 0.983                 |
| MRR@5                                                   | 0.900        | 0.939 (+0.039)       | 0.963 (+0.062)     | 0.733 (-0.167)        |
| Questions better / worse / same                         | -            | 6 / 2 / 52           | 7 / 1 / 52         | 2 / 21 / 37           |
| Added latency p50 / p95, run 1                          | 0            | 108 / 205 ms         | 214 / 262 ms       | 480 / 743 ms          |
| Added latency p50 / p95, run 2 (same code, same laptop) | 0            | 406 / 826 ms         | 269 / 313 ms       | 1703 / 4881 ms        |
| Peak VRAM                                               | -            | 1.2 GB               | 1.4 GB             | 3.1 GB                |

- Accuracy is deterministic (identical in both runs). Latency is not: the second run was 2-4x slower for bge and Qwen3 on the same laptop GPU. bge passes the p95 <= 500 ms bar in run 1 (205 ms) but not in run 2 (826 ms); jina passes in both (262 and 313 ms). Both clear the MRR bar (gain >= 0.03). Jina is 2 questions better than bge in net terms, which is within noise at n = 60.
- bge is chosen as primary mainly because of the license: jina-reranker-v3 is CC BY-NC 4.0, which blocks production use without a commercial agreement, while bge is permissive (verify on its model card). bge also uses a little less VRAM (1.2 vs 1.4 GB). **The latency advantage over jina is not established**: in run 1 bge was about half jina's latency, in run 2 it was slower. If the license is cleared, jina is the safer pick on speed and quality. Repeat the latency measurement on a quiet machine (nothing else on the GPU, laptop plugged in, 3 runs, take the median) before quoting a number.
- Qwen3-Reranker-0.6B made rankings worse (21 questions worse vs 2 better), is the slowest, and uses the most VRAM. Rejected.
- **Honest size of the win:** reranking moves one more question into the Top 5 (Hit@5 0.967 -> 0.983, 58 -> 59 of 60) and improves the position of the correct chunk (MRR). If the generator reads all 5 chunks equally, most of the benefit is ordering, and skipping the reranker is also defensible. Rerank if the order matters downstream (context truncation, showing the top 1-3 sources, citations); otherwise the extra ~100-400 ms and 1.2 GB VRAM buy about one question per 60.
- Re-test on a larger question set before treating this as proven.

## Candidates

| #   | Model                   | Type                               | Hosting         | License                                                    | Status                                                                                                                                                                                            |
| --- | ----------------------- | ---------------------------------- | --------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | BAAI/bge-reranker-v2-m3 | Cross-encoder (~0.6B)              | Self-host       | Verify on model card (sources conflict: MIT vs Apache 2.0) | Tested. **Selected**                                                                                                                                                                              |
| 2   | Qwen3-Reranker-0.6B     | LLM-based pointwise (yes/no score) | Self-host       | Apache 2.0 (verify)                                        | Tested. Rejected. The 4B size was dropped (6 GB GPU)                                                                                                                                              |
| 3   | Cohere Rerank 4         | Managed API                        | API             | Commercial                                                 | **Not tested**: paid external API, and the data is sensitive, so no data may leave the machine                                                                                                    |
| 4   | jina-reranker-v3        | Listwise                           | Self-host / API | CC BY-NC 4.0 (non-commercial)                              | Tested. Best quality, license blocks production                                                                                                                                                   |
| 5   | mxbai-rerank-v2 (base)  | Cross-encoder                      | Self-host       | Apache 2.0 (verify)                                        | **Not tested**: `mxbai_rerank` fails with transformers 5.x (`Qwen2Tokenizer has no attribute prepare_for_model`). Revisit when the library is fixed or with a pinned transformers 4.x environment |

## Setup

- Corpus and questions: the real set from the embedding benchmark, 405 chunks and 70 questions. 60 questions have a correct chunk (10 each: exact, paraphrased, short, long, technical, similar-document). 10 are no-answer questions and are excluded from retrieval metrics.
- Chunk text embedded as `"{doc_title} - {section}\n{text}"`, matching the embedding benchmark.
- Embedding model / vector store: BAAI/bge-m3 + ChromaDB (cosine), fp16 on GPU. Candidates: 20. Final: 5.
- Hardware: NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM.
- Privacy: data is sensitive, so everything ran locally with no external API. Only aggregate results are shared.
- Latency is the added time per query on top of vector search (about 1.8 ms p50), for 20 candidates, after warm-up. VRAM is peak torch allocation.
- Decision rule set before running: MRR@5 gain >= 0.03 and added p95 latency <= 500 ms.

## Results

Retrieval (60 scored questions):

| Pipeline                     | Hit@5 | Recall@5 | MRR@5 | MRR change | Better / worse / same | Added p50 (ms)        | Added p95 (ms)        | Peak VRAM (MB) | Verdict                                      |
| ---------------------------- | ----- | -------- | ----- | ---------- | --------------------- | --------------------- | --------------------- | -------------- | -------------------------------------------- |
| Vector Top 5 (baseline)      | 0.967 | 0.967    | 0.900 | -          | -                     | 0                     | 0                     | -              | -                                            |
| Top 20 + bge-reranker-v2-m3  | 0.983 | 0.983    | 0.939 | +0.039     | 6 / 2 / 52            | 107.7 (run 2: 406.4)  | 204.6 (run 2: 826.0)  | 1210           | worth it in run 1; p95 over the bar in run 2 |
| Top 20 + qwen3-reranker-0.6b | 0.983 | 0.983    | 0.733 | -0.167     | 2 / 21 / 37           | 480.3 (run 2: 1703.2) | 742.6 (run 2: 4881.2) | 3129           | worse than baseline                          |
| Top 20 + jina-reranker-v3    | 0.983 | 0.983    | 0.963 | +0.062     | 7 / 1 / 52            | 214.3 (run 2: 268.7)  | 262.3 (run 2: 312.7)  | 1389           | worth it in both runs                        |

Hit@5 / MRR@5 by question type (n = 10 each):

| Type             | Vector Top 5 | bge-reranker-v2-m3 | qwen3-reranker-0.6b | jina-reranker-v3 |
| ---------------- | ------------ | ------------------ | ------------------- | ---------------- |
| exact            | 1.00 / 0.95  | 1.00 / 1.00        | 1.00 / 0.80         | 1.00 / 1.00      |
| long             | 1.00 / 0.95  | 1.00 / 1.00        | 1.00 / 0.64         | 1.00 / 1.00      |
| paraphrased      | 1.00 / 0.83  | 1.00 / 0.90        | 1.00 / 0.88         | 1.00 / 1.00      |
| short            | 1.00 / 1.00  | 1.00 / 1.00        | 1.00 / 0.95         | 1.00 / 1.00      |
| similar-document | 1.00 / 0.87  | 1.00 / 0.88        | 1.00 / 0.55         | 1.00 / 0.88      |
| technical        | 0.80 / 0.80  | 0.90 / 0.85        | 0.90 / 0.57         | 0.90 / 0.90      |

Headroom: Top 20 contains a correct chunk for 98.3% of questions, against 96.7% for Top 5. The most Hit@5 any reranker could add is +1.7% (one question). All three rerankers reached that ceiling, so they differ only in ordering (MRR), speed and memory.

Where it helps: paraphrased, long and exact questions (correct chunk moves to rank 1) and the one technical question rescued from outside the Top 5. Similar-document questions barely improve (0.87 -> 0.88): reranking does not solve near-duplicate documents.

Synthetic check (13 questions, 42 chunks): all three rerankers fixed the same single paraphrased question (MRR 0.962 -> 1.000). This set is too small to separate them and was only used to test the pipeline.

## Limitations

- **Final RAG answer quality was not measured.** Generating answers would have required sending text to an external LLM, which was ruled out for the sensitive data. The conclusion rests on retrieval metrics only. Because the Top 5 already contains the answer for 96.7% of questions, answer quality is capped by retrieval and likely differs little between pipelines.
- n = 60 (10 per type). Differences of a few hundredths in MRR, and the bge-vs-jina gap, can be noise.
- Cohere and mxbai were not tested (reasons in the candidates table). Qwen3-Reranker was run without a task-specific instruction (default instruction), which may understate it. It is also the 0.6B size only.
- No-answer questions were not evaluated: a reranker's score cannot tell the system when to abstain, so that needs a separate score threshold or the generator to handle it.
- Licenses marked "verify" must be checked on each model card before any production use.

## Reproduce

```bash
python rerank_bench.py --config config.real.yaml --models bge-reranker-v2-m3,jina-reranker-v3,qwen3-reranker-0.6b
python make_report.py   # rebuilds results/results.md
```

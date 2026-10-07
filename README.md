# Reranker benchmark (HRD Intelligence, Task 4)

Does reranking improve retrieval enough to justify the extra compute?

- Baseline: Vector Search -> Top 5
- Reranked: Vector Search -> Top 20 -> Reranker -> Top 5

Result and decision: see [`reranker-benchmark.md`](reranker-benchmark.md).

## Layout

```
.
├── README.md
├── reranker-benchmark.md        final report and decision
├── requirements.txt
├── configs/
│   ├── config.real.yaml         real data (sensitive), runs fully local
│   ├── config.synthetic.yaml    synthetic data, safe to share
│   └── config.example.yaml      generic template
├── examples/
│   ├── corpus.example.json      corpus format
│   └── questions.example.json   questions format
├── src/
│   ├── rerank_bench.py          the benchmark (BGE-M3 + ChromaDB, vector Top 5 vs Top 20 -> rerank -> Top 5)
│   ├── make_report.py           rebuilds results/results.md from summary.json and meta.json
│   └── make_fake_data.py        fake Excel data for a --mock smoke test
├── results/                     real-data results (only results.md and summary.csv are committed)
├── results_synthetic/           synthetic results (same rule)
├── old/                         earlier template script, kept for history
└── real-data/                   sensitive input (git-ignored)
```

## Run

Run everything from the project root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# real data (local only)
python src/rerank_bench.py --config configs/config.real.yaml --models bge-reranker-v2-m3,jina-reranker-v3,qwen3-reranker-0.6b
python src/make_report.py --config configs/config.real.yaml    # rebuilds results/results.md

# synthetic data
python src/rerank_bench.py --config configs/config.synthetic.yaml --models bge-reranker-v2-m3,jina-reranker-v3,qwen3-reranker-0.6b

# wiring test without models
python src/make_fake_data.py && python src/rerank_bench.py --config fake/config.yaml --mock
```

Input files (corpus and questions) can be JSON, JSONL, Excel or CSV. Questions with an empty `expected_chunks` list are
no-answer questions and are left out of the retrieval metrics.

## Metrics

Hit@5, Recall@5, MRR@5, Top-20 ceiling (Hit@20), per-question better / worse / same vs vector-only, added latency
(p50 / p95), peak VRAM, and a per-question-type breakdown.

## Privacy

The real data is sensitive.

- `real-data/`, `real_data/`, `*.jsonl`, spreadsheets and per-query result files are git-ignored.
- Only `results/results.md` and `results/summary.csv` are committed: aggregate numbers, no document text.
- Anything that sends data outside the machine (Cohere) is off unless you pass `--allow-external`. Never use it with `config.real.yaml`.

## Not tested

- Cohere Rerank 4: paid external API, and the data may not leave the machine.
- mxbai-rerank-v2: `mxbai_rerank` fails with transformers 5.x (`prepare_for_model`). Retry with a pinned 4.x environment.

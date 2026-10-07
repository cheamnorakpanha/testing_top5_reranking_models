# Reranker benchmark harness

Compares `Vector Search -> Top 5` with `Vector Search -> Top 20 -> Reranker -> Top 5`
on your own Excel documents. Runs locally. Nothing leaves your machine unless you pass `--allow-external`.

## 1. Install (on your machine, with the RTX 3060)

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build; match your driver
pip install -r requirements.txt
```

## 2. Try it on fake data first (no GPU, no downloads)

```
python make_fake_data.py --out fake
python rerank_bench.py --config fake/config.yaml --mock
```

`--mock` swaps in toy models, so those metric values mean nothing. It only proves the pipeline runs.

## 3. Run on your real data

1. Put the Excel files in `./docs` (keeps your Attendance/ and Monthly Result/ folders).
2. Put your LLM-generated benchmark questions in `queries.xlsx` (or .csv) with two columns,
   a question and its correct chunk. Set the column names in `config.yaml`.
3. **Chunk IDs.** Each Excel row becomes one chunk with ID `<file name>::<sheet name>::<row number>`,
   where the row number counts data rows after the header, starting at 0 and skipping fully empty rows.
   Example: `Attendance_List_Aug.xlsx::Sheet1::12`.
   If your labels are the row text instead of IDs, set `label_type: text`.
4. If your sheets have title rows above the real headers, change `corpus.header_row`.
5. Run:

```
python rerank_bench.py --config config.yaml                    # local models only
python rerank_bench.py --config config.yaml --models bge-reranker-v2-m3,qwen3-reranker-0.6b
set COHERE_API_KEY=...   (or export on Linux/macOS)
python rerank_bench.py --config config.yaml --allow-external   # only if policy allows sending data to Cohere
```

## 4. What you get (in `results/`)

- `summary.csv` / `summary.json`: one row per pipeline, with `hit@5`, `recall@5`, `mrr@5`,
  `candidate_recall@20` (the ceiling a reranker can reach), added latency p50/p95 in ms, peak VRAM in MB.
  Contains no document text, so it is safe to paste back to Claude.
- `per_query_<model>.jsonl`: chunk IDs per query. Stays local. Used by the later RAG-quality step.

## Notes and known gaps

- 6 GB VRAM: all local models should fit in fp16, one at a time (the script frees each before loading the next).
  Qwen3-Reranker-4B is excluded.
- mxbai and jina loaders follow their documented APIs from memory. Check each model card if one errors out;
  a failing model is logged as an error row and the run continues.
- Cohere model string `rerank-v4.0-fast` should be verified in Cohere's docs. Latency excludes the rate-limit sleep.
- The Qwen3 model is the community `tomaarsen/...-seq-cls` conversion so it works with CrossEncoder. Confirm it
  matches the official Qwen3-Reranker scores on a few examples before trusting it.
- `.xls` (old format) needs `xlrd`; untested here. `.xlsx` was tested.
- ChromaDB was not installable in my sandbox, so the Chroma code path is untested; `--mock` used a numpy index.
  The Chroma calls are `create_collection`, `add` and `query`, so errors there should be easy to spot.

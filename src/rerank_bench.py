#!/usr/bin/env python3
"""Reranker benchmark.

Compares   Vector Search -> Top 5
against   Vector Search -> Top 20 -> Reranker -> Top 5

Everything runs locally except rerankers marked `external: true`, which are
skipped unless --allow-external is passed.

Outputs (in output_dir):
  summary.json / summary.csv   aggregate numbers only (safe to share)
  per_query_<name>.jsonl       chunk IDs per query (stay local; used for the RAG-quality step)
"""
import argparse
import csv
import gc
import glob
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def read_records(path, sheet=0):
    """Read a .json / .jsonl / .xlsx / .xls / .csv file into a list of dict records."""
    pl = path.lower()
    if pl.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        if isinstance(d, dict):  # tolerate {"chunks": [...]} style wrappers
            lists = [v for v in d.values() if isinstance(v, list)]
            d = lists[0] if lists else [d]
        return d
    if pl.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if pl.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str, sheet_name=sheet)
    else:
        df = pd.read_csv(path, dtype=str)
    return df.fillna("").to_dict("records")


def load_corpus_file(cfg):
    """Corpus given as a file of chunk records (JSON, JSONL, Excel or CSV)."""
    c = cfg["corpus"]
    recs = read_records(c["file"], c.get("sheet", 0))
    idf, tf = c.get("id_field", "chunk_id"), c.get("text_field", "text")
    prefix = c.get("prefix_fields", []) or []
    if not recs:
        raise SystemExit(f"No records found in {c['file']}")
    for needed in [idf, tf] + list(prefix):
        if needed not in recs[0]:
            raise SystemExit(
                f"Field '{needed}' not found in {c['file']}. Fields present: {list(recs[0].keys())}")
    chunks, seen = [], set()
    for r in recs:
        cid, text = str(r[idf]).strip(), str(r[tf]).strip()
        if not cid or not text:
            continue
        if cid in seen:
            print(f"[warn] duplicate chunk id {cid}: keeping the first")
            continue
        seen.add(cid)
        pre = c.get("prefix_sep", " | ").join(
            str(r[p_]).strip() for p_ in prefix if str(r.get(p_, "")).strip())
        chunks.append({"id": cid, "text": (pre + "\n" if pre else "") + text,
                       "file": str(r.get("doc_id", "")), "sheet": "", "row": len(chunks)})
    if c.get("max_chunks"):
        chunks = chunks[: int(c["max_chunks"])]
    return chunks


def build_chunks(cfg):
    c = cfg["corpus"]
    if c.get("file"):
        return load_corpus_file(cfg)
    files = []
    for pat in c["globs"]:
        files += glob.glob(os.path.join(c["dir"], pat), recursive=True)
    chunks = []
    for f in sorted(set(files)):
        if os.path.basename(f).startswith("~$"):  # Excel lock files
            continue
        try:
            sheets = pd.read_excel(
                f, sheet_name=None, header=c.get("header_row", 0), dtype=str)
        except Exception as e:  # unreadable file: report and continue
            print(f"[warn] skipping {f}: {e}")
            continue
        for sname, df in sheets.items():
            df = df.dropna(how="all").fillna("")
            cols = []
            for j, col in enumerate(df.columns):
                col = str(col).strip()
                cols.append(
                    f"col{j + 1}" if col.lower().startswith("unnamed") else col)
            for i, row in enumerate(df.itertuples(index=False)):
                parts = [f"{col}: {str(v).strip()}" for col, v in zip(
                    cols, row) if str(v).strip()]
                if not parts:
                    continue
                text = f"File: {Path(f).stem} | Sheet: {sname} | " + \
                    "; ".join(parts)
                chunks.append({"id": f"{Path(f).name}::{sname}::{i}", "text": text,
                               "file": Path(f).name, "sheet": str(sname), "row": i})
    if c.get("max_chunks"):
        chunks = chunks[: int(c["max_chunks"])]
    return chunks


def load_queries(cfg, chunks):
    """Returns (usable_queries, no_answer_queries).

    label_type options:
      chunk_ids  label_col holds the correct chunk ID(s) (list in JSON, or "c001;c002" in Excel/CSV).
                 Questions with NO correct chunk are "no-answer" questions: counted, excluded from metrics.
      id         label_col holds our row-chunk IDs (file::sheet::row)
      text       label_col holds the source text; matched by word overlap
      file       label_col holds a file name; every chunk of that file counts as correct
      row        build the chunk from file_col (+ sheet_col) and row_col
      answer     label_col holds a reference answer; chunks containing it count as correct (approximate)
    Optional type_col gives a question type (exact / paraphrased / ...) for a per-type breakdown.
    """
    q = cfg["queries"]
    path = q["file"]
    if path.lower().endswith((".json", ".jsonl")):
        df = pd.DataFrame(read_records(path))
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str, sheet_name=q.get("sheet", 0))
    else:
        df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    ltype = q.get("label_type", "id")
    sep = q.get("label_sep", ";")
    type_col = q.get("type_col")
    id_set = {c["id"] for c in chunks}
    norm_chunks = [(c["id"], norm(c["text"])) for c in chunks]
    chunk_tokens = [(c["id"], set(re.findall(r"\w+", c["text"].lower())))
                    for c in chunks]
    by_file, by_file_row = {}, {}
    for c in chunks:
        stem = Path(c["file"]).stem.lower()
        by_file.setdefault(c["file"].lower(), []).append(c["id"])
        by_file.setdefault(stem, []).append(c["id"])
        by_file_row.setdefault((c["file"].lower(), c["row"]), []).append(
            (c["sheet"], c["id"]))
    for needed in [q["query_col"]] + ([q["label_col"]] if ltype != "row" else [q["file_col"], q["row_col"]]):
        if needed not in df.columns:
            raise SystemExit(
                f"Column/key '{needed}' not found in {path}. Present: {list(df.columns)}")

    out, skipped, no_answer, unknown_ids = [], 0, 0, 0
    na_list = []
    for qi, r in df.iterrows():
        query = str(r[q["query_col"]]).strip()
        qtype = (str(r[type_col]).strip()
                 or "all") if type_col and type_col in df.columns else "all"
        rel = []
        if ltype == "chunk_ids":
            val = r[q["label_col"]]
            if isinstance(val, (list, tuple)):
                labels = [str(x).strip() for x in val if str(x).strip()]
            else:
                sv = str(val).strip()
                try:
                    parsed = json.loads(sv) if sv.startswith("[") else None
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    labels = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    labels = [x.strip() for x in re.split(r"[;,|]", sv.strip(
                        "[]").replace("'", "").replace('"', "")) if x.strip()]
            if not labels:
                no_answer += 1
                if query:
                    na_list.append(
                        {"qid": int(qi), "query": query, "qtype": qtype})
                continue
            rel = [l for l in labels if l in id_set]
            unknown_ids += len(labels) - len(rel)
        elif ltype == "row":
            fname = str(r[q["file_col"]]).strip().lower()
            try:
                rown = int(float(str(r[q["row_col"]]).strip())
                           ) + int(q.get("row_offset", 0))
            except ValueError:
                rown = None
            if rown is not None:
                for sheet, cid in by_file_row.get((fname, rown), []):
                    sc = q.get("sheet_col")
                    if not sc or str(r[sc]).strip() == sheet:
                        rel.append(cid)
        else:
            raw = str(r[q["label_col"]]).strip()
            # text/answer labels may contain the separator themselves: never split them
            labels = [x.strip() for x in raw.split(sep) if x.strip()] if ltype in (
                "id", "file") else ([raw] if raw else [])
            if ltype == "id":
                rel = [l for l in labels if l in id_set]
            elif ltype == "file":
                for l in labels:
                    rel += by_file.get(l.lower(),
                                       by_file.get(Path(l).stem.lower(), []))
                rel = list(dict.fromkeys(rel))
            elif ltype == "text":
                lt = set(re.findall(
                    r"\w+", labels[0].lower())) if labels else set()
                if lt:
                    best, best_ids = 0.0, []
                    for cid, ct in chunk_tokens:
                        cov = len(lt & ct) / len(lt)
                        if cov > best + 1e-9:
                            best, best_ids = cov, [cid]
                        elif abs(cov - best) <= 1e-9:
                            best_ids.append(cid)
                    if best >= float(q.get("min_token_coverage", 0.8)):
                        rel = best_ids
            elif ltype == "answer":
                nl = [norm(l) for l in labels]
                rel = [cid for cid, t in norm_chunks if any(
                    l and l in t for l in nl)]
            else:
                raise SystemExit(f"Unknown label_type: {ltype}")
        if not query or not rel:
            skipped += 1
            continue
        out.append({"qid": int(qi), "query": query,
                   "relevant": rel, "qtype": qtype})
    avg_rel = (sum(len(o["relevant"]) for o in out) / len(out)) if out else 0
    types = {}
    for o in out:
        types[o["qtype"]] = types.get(o["qtype"], 0) + 1
    print(f"[data] {len(chunks)} chunks, {len(out)} usable queries, {skipped} skipped (no match), "
          f"avg {avg_rel:.1f} correct chunks per query")
    if len(types) > 1 or "all" not in types:
        print(f"[data] question types: {types}")
    if no_answer:
        print(
            f"[data] {no_answer} no-answer questions (no correct chunk): excluded from retrieval metrics")
    if unknown_ids:
        print(
            f"[warn] {unknown_ids} labeled chunk IDs were not found in the corpus: check corpus.id_field / ID formats")
    if ltype in ("file", "answer") and avg_rel > 5:
        print("[warn] many chunks count as correct per query; Recall@5 will be low by construction, "
              "so compare models by Hit@5 and MRR@5.")
    if not out:
        raise SystemExit(
            "No usable queries. Check queries.file / column names / label_type in config.yaml")
    return out, na_list


# ----------------------------------------------------------------------------
# Embedders
# ----------------------------------------------------------------------------
class MockEmbedder:
    """Hashed bag-of-words. For testing the pipeline only (--mock)."""
    dim = 256

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) %
                    self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


class STEmbedder:
    def __init__(self, name, device):
        import torch
        from sentence_transformers import SentenceTransformer
        kw = {"torch_dtype": torch.float16} if device.startswith("cuda") else {
        }
        self.m = SentenceTransformer(name, device=device, model_kwargs=kw)

    def encode(self, texts, batch_size=16):
        return self.m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=False)


class NumpyIndex:
    """Brute-force cosine index mimicking the one Chroma call we use. Test mode only."""

    def __init__(self, ids, vecs):
        self.ids, self.vecs = ids, np.asarray(vecs, dtype=np.float32)

    def query(self, query_embeddings, n_results):
        sims = self.vecs @ np.asarray(query_embeddings[0], dtype=np.float32)
        order = np.argsort(-sims, kind="stable")[:n_results]
        return {"ids": [[self.ids[i] for i in order]]}


# ----------------------------------------------------------------------------
# Rerankers. Each returns document indices sorted best-first.
# ----------------------------------------------------------------------------
class MockReranker:
    """Token-overlap scorer. For testing the pipeline only (--mock)."""

    def rank(self, query, docs):
        qt = set(re.findall(r"\w+", query.lower()))
        sc = [len(qt & set(re.findall(r"\w+", d.lower()))) /
              (len(qt) + 1e-9) for d in docs]
        return list(np.argsort(-np.asarray(sc), kind="stable"))

    def close(self):
        pass


class CrossEncoderReranker:
    def __init__(self, model_id, device, max_length=512, batch_size=16):
        import torch
        from sentence_transformers import CrossEncoder
        kw = {"torch_dtype": torch.float16} if device.startswith("cuda") else {
        }
        self.bs = batch_size
        self.m = CrossEncoder(model_id, device=device,
                              max_length=max_length, model_kwargs=kw)

    def rank(self, query, docs):
        scores = self.m.predict([(query, d) for d in docs], batch_size=self.bs,
                                show_progress_bar=False)
        return list(np.argsort(-np.asarray(scores), kind="stable"))

    def close(self):
        del self.m


class Qwen3Reranker:
    """Official Qwen3-Reranker usage: relevance = P("yes") under a fixed chat prompt.
    It is NOT a plain cross-encoder, so scoring it through CrossEncoder gives bad rankings."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, model_id, device, instruction=None, batch_size=8, max_length=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        kw = {"torch_dtype": torch.float16} if device.startswith("cuda") else {
        }
        self.m = AutoModelForCausalLM.from_pretrained(
            model_id, **kw).to(device).eval()
        self.device, self.bs, self.max_length = device, batch_size, max_length
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(
            self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(
            self.SUFFIX, add_special_tokens=False)

    @staticmethod
    def format_pair(instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def rank(self, query, docs):
        torch = self.torch
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        scores = []
        for s in range(0, len(docs), self.bs):
            texts = [self.format_pair(self.instruction, query, d)
                     for d in docs[s:s + self.bs]]
            enc = self.tok(texts, padding=False, truncation="longest_first",
                           return_attention_mask=False, max_length=budget)
            enc["input_ids"] = [self.prefix_ids + ids +
                                self.suffix_ids for ids in enc["input_ids"]]
            enc = self.tok.pad(enc, padding=True,
                               return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.m(**enc).logits[:, -1, :]
            pair = torch.stack(
                [logits[:, self.no], logits[:, self.yes]], dim=1).float()
            scores += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return list(np.argsort(-np.asarray(scores), kind="stable"))

    def close(self):
        del self.m


class MxbaiReranker:
    # VERIFY against the model card: constructor kwargs / device handling may differ by version.
    def __init__(self, model_id, device):
        from mxbai_rerank import MxbaiRerankV2
        self.m = MxbaiRerankV2(model_id)

    def rank(self, query, docs):
        res = self.m.rank(query, docs, return_documents=False, top_k=len(docs))
        return [r["index"] if isinstance(r, dict) else r.index for r in res]

    def close(self):
        del self.m


class JinaReranker:
    # VERIFY against the model card: the rerank() signature may differ by version.
    def __init__(self, model_id, device):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, dtype="auto")
        self.m.eval()
        self.m.to(device)

    def rank(self, query, docs):
        res = self.m.rerank(query, docs, top_n=len(docs))
        return [r["index"] for r in res]

    def close(self):
        del self.m


class CohereReranker:
    def __init__(self, model_id, sleep_s=0):
        import cohere
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set the COHERE_API_KEY environment variable")
        self.c = cohere.ClientV2(key)
        self.model_id, self.sleep_s = model_id, sleep_s

    def rank(self, query, docs):
        if self.sleep_s:
            # sleep is excluded from latency in timed_rank()
            time.sleep(self.sleep_s)
        r = self.c.rerank(model=self.model_id, query=query,
                          documents=docs, top_n=len(docs))
        return [x.index for x in r.results]

    def close(self):
        pass


def make_reranker(spec, device, mock):
    if mock:
        return MockReranker()
    kind = spec["kind"]
    if kind == "cross_encoder":
        return CrossEncoderReranker(spec["model_id"], device)
    if kind == "qwen3":
        return Qwen3Reranker(spec["model_id"], device, spec.get("instruction"),
                             spec.get("batch_size", 8), spec.get("max_length", 1024))
    if kind == "mxbai":
        return MxbaiReranker(spec["model_id"], device)
    if kind == "jina":
        return JinaReranker(spec["model_id"], device)
    if kind == "cohere":
        return CohereReranker(spec["model_id"], spec.get("sleep_s", 0))
    raise ValueError(f"unknown reranker kind: {kind}")


# ----------------------------------------------------------------------------
# GPU helpers and metrics
# ----------------------------------------------------------------------------
def cuda_ok(device):
    if not device.startswith("cuda"):
        return False
    import torch
    return torch.cuda.is_available()


def sync(device):
    if cuda_ok(device):
        import torch
        torch.cuda.synchronize()


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def by_type(ranked_lists, queries, k):
    out = {}
    for t in sorted({q["qtype"] for q in queries}):
        idx = [i for i, q in enumerate(queries) if q["qtype"] == t]
        m = score([ranked_lists[i] for i in idx], [
                  queries[i]["relevant"] for i in idx], k)
        m["n"] = len(idx)
        out[t] = m
    return out


def score(ranked_lists, relevant_lists, k):
    hit, rec, rr = [], [], []
    for ranked, rel in zip(ranked_lists, relevant_lists):
        top, rs = ranked[:k], set(rel)
        hit.append(1.0 if any(i in rs for i in top) else 0.0)
        rec.append(len(rs & set(top)) / len(rs))
        r_ = 0.0
        for pos, i in enumerate(top, 1):
            if i in rs:
                r_ = 1.0 / pos
                break
        rr.append(r_)
    return {f"hit@{k}": round(statistics.mean(hit), 4),
            f"recall@{k}": round(statistics.mean(rec), 4),
            f"mrr@{k}": round(statistics.mean(rr), 4)}


# ----------------------------------------------------------------------------
# Results report (markdown) and merging of partial runs
# ----------------------------------------------------------------------------
def merge_with_previous(out_dir, summary, meta, order):
    """Keep results of earlier runs (e.g. a different --models subset) if they used the same data."""
    sp, mp = out_dir / "summary.json", out_dir / "meta.json"
    if sp.exists() and mp.exists():
        try:
            old_meta = json.loads(mp.read_text(encoding="utf-8"))
            old = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return summary
        same = all(old_meta.get(k) == meta.get(k)
                   for k in ("n_chunks", "n_queries", "label_type", "embedding"))
        if same:
            by = {r["reranker"]: r for r in old}
            by.update({r["reranker"]: r for r in summary})
            names = ["-"] + [n for n in order if n in by]
            names += [n for n in by if n not in names]
            kept = [n for n in by if n not in [r["reranker"] for r in summary]]
            if kept:
                print(
                    f"[merge] kept earlier results for: {', '.join(n for n in kept if n != '-')}")
            return [by[n] for n in names]
        print(
            "[merge] data or settings changed since the last run: starting a fresh summary")
    return summary


def paired_counts(pq_dir, name, k):
    """Compare a reranker with the vector-only baseline per query, from the per_query_*.jsonl files.
    Returns (better, worse, same) in reciprocal-rank terms, or None if the files are missing."""
    def load(fname):
        fp = Path(pq_dir) / fname
        if not fp.exists():
            return None
        return {r["qid"]: r for r in (json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip())}

    def rr(rec):
        rs = set(rec["relevant"])
        for pos, i in enumerate(rec["top"][:k], 1):
            if i in rs:
                return 1.0 / pos
        return 0.0

    base, mod = load("per_query_vector_only.jsonl"), load(
        f"per_query_{name}.jsonl")
    if not base or not mod:
        return None
    better = worse = same = 0
    for qid, rec in mod.items():
        if rec.get("no_answer") or qid not in base:
            continue
        d = rr(rec) - rr(base[qid])
        if d > 1e-9:
            better += 1
        elif d < -1e-9:
            worse += 1
        else:
            same += 1
    return better, worse, same


def write_markdown(path, summary, meta, cfg):
    k, n = cfg["retrieval"]["final_k"], cfg["retrieval"]["candidates"]
    dec = cfg.get("decision", {}) or {}
    min_gain = float(dec.get("min_mrr_gain", 0.03))
    max_p95 = dec.get("max_added_p95_ms", 500)
    external = {r["name"] for r in cfg["rerankers"] if r.get("external")}
    base = next((r for r in summary if r.get(
        "pipeline") == "vector_only"), None)
    rows = [r for r in summary if str(r.get("pipeline", "")).startswith("top")]
    errs = [r for r in summary if r.get("pipeline") == "ERROR"]
    if base is None:
        return
    hk, rk, mk = f"hit@{k}", f"recall@{k}", f"mrr@{k}"
    ceil_hit = base.get(f"candidate_hit@{n}")

    def verdict(r):
        gain = r[mk] - base[mk]
        if gain < min_gain:
            return "gain too small"
        if max_p95 is not None and r["added_ms_p95"] > float(max_p95):
            return "too slow"
        return "worth it"

    L = []
    L.append("# Reranker benchmark results")
    L.append("")
    L.append(
        f"_Generated {meta['date']}. Contains aggregate metrics only, no document text._")
    L.append("")
    L.append("## Run setup")
    L.append("")
    L.append(f"- Chunks: {meta['n_chunks']} ({meta.get('corpus_desc', 'one per spreadsheet row')}) | Queries: {meta['n_queries']} "
             f"(avg {meta['avg_relevant']:.1f} correct chunk(s) per query, label type `{meta['label_type']}`)")
    if meta.get("n_no_answer"):
        L.append(
            f"- No-answer questions: {meta['n_no_answer']} (no correct chunk exists; excluded from retrieval metrics)")
    L.append(
        f"- Embedding: {meta['embedding']} + ChromaDB | Device: {meta['device']}")
    L.append(
        f"- Pipelines: Vector Top {k} vs Vector Top {n} -> reranker -> Top {k}")
    L.append("")
    L.append("## Retrieval results")
    L.append("")
    L.append(
        f"| Pipeline | Hit@{k} | Recall@{k} | MRR@{k} | MRR change | Questions better / worse / same | Added latency p50 (ms) | p95 (ms) | Peak VRAM (MB) | Verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    L.append(
        f"| Vector Top {k} (baseline) | {base[hk]:.3f} | {base[rk]:.3f} | {base[mk]:.3f} | - | - | 0 | 0 | - | - |")
    for r in rows:
        vram = "n/a (API)" if r["reranker"] in external else (
            str(r["vram_mb_peak"]) if r["vram_mb_peak"] else "n/a")
        pc = paired_counts(Path(path).parent, r["reranker"], k)
        pcs = f"{pc[0]} / {pc[1]} / {pc[2]}" if pc else "n/a"
        L.append(f"| Top {n} + {r['reranker']} | {r[hk]:.3f} | {r[rk]:.3f} | {r[mk]:.3f} | "
                 f"{r[mk] - base[mk]:+.3f} | {pcs} | {r['added_ms_p50']} | {r['added_ms_p95']} | {vram} | {verdict(r)} |")
    L.append("")
    L.append(f"Baseline vector search latency: p50 {base['retrieval_ms_p50']} ms, p95 {base['retrieval_ms_p95']} ms. "
             f"Reranker latency is the extra time on top of that, for {n} candidates per query.")
    L.append("")
    types = sorted((base.get("by_type") or {}).keys())
    if len(types) > 1:
        L.append(f"## Results by question type")
        L.append("")
        L.append(
            f"Hit@{k} / MRR@{k} per question type (n = number of questions of that type).")
        L.append("")
        L.append("| Type | n | Vector Top " + str(k) + " |" +
                 "".join(f" {r['reranker']} |" for r in rows))
        L.append("|---|---|---|" + "---|" * len(rows))
        for t in types:
            m = base["by_type"][t]
            cells = [str(m["n"]), f"{m[hk]:.2f} / {m[mk]:.2f}"]
            for r in rows:
                mm = (r.get("by_type") or {}).get(t)
                cells.append(f"{mm[hk]:.2f} / {mm[mk]:.2f}" if mm else "-")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Headroom")
    L.append("")
    if ceil_hit is not None:
        L.append(f"A reranker can only reorder what vector search already retrieved. Top-{n} contains a correct chunk for "
                 f"{ceil_hit:.1%} of queries (Hit@{n}), while the baseline Top {k} already has one for {base[hk]:.1%}. "
                 f"Maximum possible Hit@{k} gain from reranking: {max(0.0, ceil_hit - base[hk]):+.1%}.")
    L.append("")
    L.append("## Decision helper (not the final decision)")
    L.append("")
    lim = f", added p95 latency <= {max_p95} ms" if max_p95 is not None else ""
    L.append(f"Rule used: MRR@{k} gain >= {min_gain}{lim}. These thresholds are assumptions; "
             f"change them under `decision:` in config.yaml and rerun the report.")
    L.append("")
    ok = [r for r in rows if verdict(r) == "worth it"]
    if ok:
        best = max(ok, key=lambda r: r[mk])
        L.append(f"- Best model meeting the rule: **{best['reranker']}** (MRR@{k} {best[mk]:.3f}, "
                 f"{best[mk] - base[mk]:+.3f} vs baseline, +{best['added_ms_p50']} ms p50).")
        if meta["n_queries"] < 200:
            L.append(f"- With only {meta['n_queries']} questions, 'worth it' means worth confirming, not proven. "
                     "Check the better / worse / same column: if the better and worse counts are close, the gain is likely noise.")
    elif rows:
        L.append("- No reranker meets the rule on retrieval metrics alone. Reasonable outcome: **skip reranking**, "
                 "unless the RAG-quality step shows a clear answer-quality gain.")
    else:
        L.append("- No reranker results yet.")
    if errs:
        L.append("")
        L.append("Models that failed to run:")
        for e in errs:
            L.append(f"- {e['reranker']}: {e.get('error', '')}")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append(
        f"- Final RAG answer quality is **not measured here**; that is the next step.")
    L.append(f"- With {meta['n_queries']} queries, differences of a few hundredths in MRR can be noise. "
             "Treat small gaps cautiously.")
    L.append("- Cohere (if present) ran through an external API: no VRAM figure, and its latency includes network time.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", default="",
                    help="comma-separated reranker names (default: all)")
    ap.add_argument("--allow-external", action="store_true",
                    help="allow rerankers marked external (data is sent to a third party)")
    ap.add_argument("--mock", action="store_true",
                    help="pipeline test with fake models, no downloads")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    n_cand, k = cfg["retrieval"]["candidates"], cfg["retrieval"]["final_k"]
    print(
        f"[env] device={device} candidates={n_cand} final_k={k} mock={args.mock}")

    chunks = build_chunks(cfg)
    queries, na_queries = load_queries(cfg, chunks)
    n_no_answer = len(na_queries)
    texts = {c["id"]: c["text"] for c in chunks}

    # ---- embed + index ----------------------------------------------------
    emb = MockEmbedder() if args.mock else STEmbedder(
        cfg["embedding"]["model"], device)
    bs = cfg["embedding"].get("batch_size", 16)
    t0 = time.time()
    chunk_vecs = emb.encode([c["text"] for c in chunks], batch_size=bs)
    query_vecs = emb.encode([q["query"] for q in queries], batch_size=bs)
    na_vecs = emb.encode([q["query"] for q in na_queries],
                         batch_size=bs) if na_queries else []
    print(f"[embed] done in {time.time() - t0:.1f}s")
    if not args.mock:
        del emb
        gc.collect()
        if cuda_ok(device):
            import torch
            torch.cuda.empty_cache()

    try:
        import chromadb
    except ImportError:
        if not args.mock:
            raise SystemExit("chromadb is not installed: pip install chromadb")
        chromadb = None
        print(
            "[warn] chromadb missing: using numpy index (allowed in --mock test mode only)")
    if chromadb is not None:
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("bench")
        except Exception:
            pass
        col = client.create_collection(
            "bench", metadata={"hnsw:space": "cosine"})
        for s in range(0, len(chunks), 2000):
            part = chunks[s:s + 2000]
            col.add(ids=[c["id"] for c in part], embeddings=chunk_vecs[s:s + 2000].tolist(),
                    documents=[c["text"] for c in part])
    else:
        col = NumpyIndex([c["id"] for c in chunks], chunk_vecs)

    # ---- first-stage retrieval -----------------------------------------------
    cands, ret_ms = [], []
    for qv in query_vecs:
        t = time.perf_counter()
        res = col.query(query_embeddings=[
                        qv.tolist()], n_results=min(n_cand, len(chunks)))
        ret_ms.append((time.perf_counter() - t) * 1000)
        cands.append(res["ids"][0])
    rel = [q["relevant"] for q in queries]
    na_cands = [col.query(query_embeddings=[v.tolist()], n_results=min(
        n_cand, len(chunks)))["ids"][0] for v in na_vecs]

    summary = []
    base = {"pipeline": "vector_only", "reranker": "-",
            **score(cands, rel, k),
            f"candidate_recall@{n_cand}": score(cands, rel, n_cand)[f"recall@{n_cand}"],
            f"candidate_hit@{n_cand}": score(cands, rel, n_cand)[f"hit@{n_cand}"],
            "retrieval_ms_p50": round(pct(ret_ms, 50), 2), "retrieval_ms_p95": round(pct(ret_ms, 95), 2),
            "added_ms_p50": 0, "added_ms_p95": 0, "vram_mb_peak": 0, "load_s": 0,
            "by_type": by_type(cands, queries, k)}
    summary.append(base)
    with open(out_dir / "per_query_vector_only.jsonl", "w", encoding="utf-8") as f:
        for q, c in zip(queries, cands):
            f.write(json.dumps(
                {"qid": q["qid"], "top": c[:k], "relevant": q["relevant"]}) + "\n")
        for q, c in zip(na_queries, na_cands):
            f.write(json.dumps(
                {"qid": q["qid"], "top": c[:k], "relevant": [], "no_answer": True}) + "\n")
    print(f"[baseline] {base}")

    # ---- rerankers -------------------------------------------------------------
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for spec in cfg["rerankers"]:
        name = spec["name"]
        if wanted and name not in wanted:
            continue
        if spec.get("external") and not args.allow_external and not args.mock:
            print(
                f"[skip] {name}: external API, pass --allow-external to send data to a third party")
            continue
        print(f"[run] {name}")
        try:
            if cuda_ok(device):
                import torch
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t_load = time.time()
            rr = make_reranker(spec, device, args.mock)
            load_s = time.time() - t_load

            def timed_rank(query, docs):
                sync(device)
                t = time.perf_counter()
                order = rr.rank(query, docs)
                sync(device)
                ms = (time.perf_counter() - t) * 1000
                ms -= (spec.get("sleep_s", 0) *
                       1000) if spec["kind"] == "cohere" and not args.mock else 0
                return order, ms

            for q, c in list(zip(queries, cands))[: cfg.get("warmup_queries", 3)]:
                timed_rank(q["query"], [texts[i] for i in c])
            if cuda_ok(device):
                import torch
                torch.cuda.reset_peak_memory_stats()

            reranked, ms = [], []
            for q, c in zip(queries, cands):
                order, t_ms = timed_rank(q["query"], [texts[i] for i in c])
                reranked.append([c[i] for i in order])
                ms.append(t_ms)
            na_reranked = []
            # not timed, not scored: only saved for the RAG-quality step
            for q, c in zip(na_queries, na_cands):
                order, _ = timed_rank(q["query"], [texts[i] for i in c])
                na_reranked.append([c[i] for i in order])
            vram = 0
            if cuda_ok(device) and not spec.get("external"):
                import torch
                vram = torch.cuda.max_memory_allocated() / 1024 ** 2
            row = {"pipeline": f"top{n_cand}_rerank_top{k}", "reranker": name,
                   **score(reranked, rel, k),
                   f"candidate_recall@{n_cand}": base[f"candidate_recall@{n_cand}"],
                   f"candidate_hit@{n_cand}": base[f"candidate_hit@{n_cand}"],
                   "retrieval_ms_p50": base["retrieval_ms_p50"], "retrieval_ms_p95": base["retrieval_ms_p95"],
                   "added_ms_p50": round(pct(ms, 50), 1), "added_ms_p95": round(pct(ms, 95), 1),
                   "vram_mb_peak": round(vram), "load_s": round(load_s, 1),
                   "by_type": by_type(reranked, queries, k)}
            summary.append(row)
            print(f"  {row}")
            with open(out_dir / f"per_query_{name}.jsonl", "w", encoding="utf-8") as f:
                for q, r in zip(queries, reranked):
                    f.write(json.dumps(
                        {"qid": q["qid"], "top": r[:k], "relevant": q["relevant"]}) + "\n")
                for q, r in zip(na_queries, na_reranked):
                    f.write(json.dumps(
                        {"qid": q["qid"], "top": r[:k], "relevant": [], "no_answer": True}) + "\n")
            rr.close()
        except Exception as e:  # one broken model must not kill the whole run
            print(f"  [error] {name}: {type(e).__name__}: {e}")
            summary.append({"pipeline": "ERROR", "reranker": name,
                           "error": f"{type(e).__name__}: {e}"})
        gc.collect()

    # ---- write summary + markdown report ---------------------------------------
    gpu = ""
    if cuda_ok(device):
        import torch
        gpu = f" ({torch.cuda.get_device_name(0)})"
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "n_chunks": len(chunks), "n_queries": len(queries),
            "avg_relevant": sum(len(q["relevant"]) for q in queries) / len(queries),
            "label_type": cfg["queries"].get("label_type", "id"), "embedding": cfg["embedding"]["model"],
            "n_no_answer": n_no_answer,
            "corpus_desc": "from corpus file" if cfg["corpus"].get("file") else "one per spreadsheet row",
            "device": device + gpu + (" [MOCK RUN: numbers are meaningless]" if args.mock else "")}
    summary = merge_with_previous(out_dir, summary, meta, [
                                  r["name"] for r in cfg["rerankers"]])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat = []
    for r in summary:
        fr = {k_: v for k_, v in r.items() if k_ != "by_type"}
        for t, m in (r.get("by_type") or {}).items():
            for mk_, mv in m.items():
                fr[f"{mk_}[{t}]"] = mv
        flat.append(fr)
    keys = sorted({k_ for r in flat for k_ in r})
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    write_markdown(out_dir / "results.md", summary, meta, cfg)
    print(
        f"[done] wrote {out_dir / 'summary.csv'} and {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()

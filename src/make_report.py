#!/usr/bin/env python3
"""Rebuild results/results.md from results/summary.json + meta.json (e.g. after editing `decision:` in configs/config.real.yaml)."""
import argparse
import json
from pathlib import Path

import yaml

import rerank_bench as rb

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.real.yaml")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
out = Path(cfg["output_dir"])
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
rb.write_markdown(out / "results.md", summary, meta, cfg)
print(f"wrote {out / 'results.md'}")

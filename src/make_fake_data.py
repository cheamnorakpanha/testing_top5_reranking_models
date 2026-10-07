#!/usr/bin/env python3
"""Generate FAKE attendance / monthly-result Excel files and a labeled query sheet.

Same folder shape as the real docs (Attendance/, Monthly Result/) but every value is invented.
Usage: python make_fake_data.py [--out fake]
"""
import argparse
import random
from pathlib import Path

import pandas as pd

FIRST = ["Sok", "Chan", "Dara", "Vanna", "Sreyleak",
         "Rith", "Kosal", "Mony", "Pich", "Leakena"]
LAST = ["Kim", "Heng", "Sao", "Lim", "Chea", "Ouk", "Meas", "Tep"]
COURSES = ["Python Basics", "Data Structures",
           "SQL and Databases", "Web Development", "Machine Learning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake")
    ap.add_argument("--students", type=int, default=40)
    args = ap.parse_args()
    random.seed(7)
    out = Path(args.out)
    (out / "docs" / "Attendance").mkdir(parents=True, exist_ok=True)
    (out / "docs" / "Monthly Result").mkdir(parents=True, exist_ok=True)

    students = [(f"STU{n:03d}", f"{random.choice(FIRST)} {random.choice(LAST)}")
                for n in range(1, args.students + 1)]
    queries = []

    def save(df, path, tag):
        df.to_excel(path, index=False, sheet_name="Sheet1")
        return Path(path).name

    # Attendance files
    for fname, month in [("Student Attendance for Aug 2026.xlsx", "Aug 2026"),
                         ("Attendance_List_Aug.xlsx", "Aug 2026 (list)")]:
        rows = [{"Student ID": sid, "Student Name": name, "Month": month,
                 "Days Present": random.randint(10, 22), "Days Absent": random.randint(0, 6),
                 "Late": random.randint(0, 5)} for sid, name in students]
        df = pd.DataFrame(rows)
        fn = save(df, out / "docs" / "Attendance" / fname, "att")
        for i in random.sample(range(len(df)), 12):
            queries.append((f"How many days was {df.loc[i, 'Student Name']} ({df.loc[i, 'Student ID']}) absent in {month}?",
                            f"{fn}::Sheet1::{i}"))

    # Monthly result files
    for fname, month in [("April Monthly Result_14th Generation.xlsx", "April"),
                         ("August Monthly Result - AI Engineering Master.xlsx", "August")]:
        rows = []
        for sid, name in students:
            rows.append({"Student ID": sid, "Student Name": name, "Month": month,
                         "Course": random.choice(COURSES), "Score": random.randint(40, 100),
                         "Grade": random.choice(list("ABCD")), "Remark": random.choice(["Pass", "Pass", "Retake"])})
        df = pd.DataFrame(rows)
        fn = save(df, out / "docs" / "Monthly Result" / fname, "res")
        for i in random.sample(range(len(df)), 12):
            queries.append((f"What was the {month} score of {df.loc[i, 'Student Name']} ({df.loc[i, 'Student ID']})?",
                            f"{fn}::Sheet1::{i}"))

    pd.DataFrame(queries, columns=["query", "relevant"]).to_excel(
        out / "queries.xlsx", index=False)

    # Self-contained test config (paths relative to the folder you run the script from)
    base = out.as_posix()
    (out / "config.yaml").write_text(f"""corpus: {{dir: ./{base}/docs, globs: ["**/*.xlsx", "**/*.xls"], header_row: 0, max_chunks: null}}
queries: {{file: ./{base}/queries.xlsx, query_col: query, label_col: relevant, label_type: id, label_sep: ";"}}
embedding: {{model: BAAI/bge-m3, batch_size: 16}}
retrieval: {{candidates: 20, final_k: 5}}
device: cpu
output_dir: ./{base}/results
warmup_queries: 2
rerankers:
  - {{name: mock-a, kind: cross_encoder, model_id: x, external: false}}
""", encoding="utf-8")
    print(
        f"wrote fake data under {out}/ ({len(queries)} queries) and {out}/config.yaml")


if __name__ == "__main__":
    main()

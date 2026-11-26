"""onset_range.py - days since stroke onset for the analysed subjects (Section 2.1). Usage: python onset_range.py"""
import csv
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
import glob
found = [p for p in glob.glob(os.path.join(ROOT, "**", "participants.tsv"), recursive=True)
         if "venv" not in p and ("stroke" in p.lower() or "figshare" in p.lower())]
if not found:
    raise SystemExit("could not find the stroke participants.tsv under the project folder")
TSV = found[0]
print("reading", TSV)

cache = np.load(os.path.join(ROOT, "cache", "clean_all_400.npz"), allow_pickle=True)
kept = set(map(str, np.unique(cache["stroke_s"])))

rows = list(csv.DictReader(open(TSV), delimiter="\t"))
print("columns in participants.tsv:", list(rows[0].keys()))
cols = [c for c in rows[0] if re.search(r"onset|day|time|duration|course", c, re.I)]
if not cols:
    raise SystemExit("no onset-like column found; pick it from the list above and edit `cols`")

for col in cols:
    vals = []
    for r in rows:
        m = re.search(r"(\d+)", r.get("Participant_ID", ""))
        sid = f"stroke_{int(m.group(1)):02d}" if m else None
        if sid in kept:
            num = re.search(r"[-+]?\d*\.?\d+", r.get(col) or "")
            if num:
                vals.append(float(num.group()))
    if vals:
        v = np.array(vals)
        q1, q3 = np.percentile(v, [25, 75])
        print(f"{col}: n={len(v)} of {len(kept)} kept subjects, range {v.min():g}-{v.max():g}, "
              f"median {np.median(v):g} (IQR {q1:g}-{q3:g})")

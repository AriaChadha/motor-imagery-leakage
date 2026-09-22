"""preprint_audit.py — pre-posting verification."""
import json, re, zipfile, glob
from pathlib import Path
import pandas as pd
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def doc_text(path):
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    return "\n".join(
        "".join(t.text or "" for t in p.iter(W + "t"))
        for p in root.iter(W + "p")
    )

docx = sorted(glob.glob("*.docx"))
print("=== docx candidates ===", docx, "\n")
text = doc_text(docx[0]) if docx else ""

print("=== stale numbers in manuscript ===")
for pat in [r"\+?0\.72", r"100(\.0)?\s*%\s*goal", r"Parkinson",
            r"ds004584", r"EEGNet", r"bigP3|ALS"]:
    hits = [m.group(0) for m in re.finditer(pat, text, re.I)]
    print(f"  {pat:30s} -> {len(hits)} hit(s) {hits[:3]}")

print("\n=== 0.72 in results.csv ===")
df = pd.read_csv("results.csv")
print("  columns:", list(df.columns))
num = df.select_dtypes("number")
mask = ((num - 0.0072).abs() < 1e-4) | ((num - 0.72).abs() < 1e-3)
print("  matches:", int(mask.to_numpy().sum()))

print("\n=== retained trials by group ===")
for col in ("group", "cohort", "dataset"):
    if col in df.columns:
        if "n_trials" in df.columns:
            print(df.groupby(col)["n_trials"].sum())
        else:
            print(df[col].value_counts())
        break
else:
    print("  no group column found")

p = Path("outputs/pipeline_results.json")
if p.exists():
    print("  json keys:", list(json.loads(p.read_text()))[:20])

print("\n=== latency provenance ===")
LAT = re.compile(r"latency", re.I)
NUM = re.compile(r"=\s*[\d.]|275|0\.275|250")
for f in sorted(Path(".").rglob("*.py")):
    if f.name == "preprint_audit.py":
        continue
    for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
        if LAT.search(line) and NUM.search(line):
            print(f"  {f}:{i}: {line.strip()[:110]}")

"""
audit_results.py — dump everything needed to resolve the open manuscript conflicts.

Makes no assumptions about column names. Run from the project root:
    python audit_results.py
Writes the full output to outputs/audit.txt as well as printing it.
"""

import os
import glob
import sys

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)

LINES = []


def out(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LINES.append(s)


def rule(title):
    out("\n" + "=" * 78)
    out(title)
    out("=" * 78)


def find_csvs():
    hits = []
    for pat in ("results*.csv", "outputs/results*.csv", "**/results*.csv"):
        hits += glob.glob(pat, recursive=True)
    seen, keep = set(), []
    for h in hits:
        rp = os.path.realpath(h)
        if rp in seen or "venv" in h or "site-packages" in h:
            continue
        seen.add(rp)
        keep.append(h)
    return keep


def main():
    rule("CSV FILES FOUND")
    csvs = find_csvs()
    if not csvs:
        out("No results*.csv found. Listing every csv in the tree instead:")
        csvs = [c for c in glob.glob("**/*.csv", recursive=True)
                if "venv" not in c and "site-packages" not in c]
        for c in csvs:
            out("   ", c, f"({os.path.getsize(c)} bytes)")
        if not csvs:
            out("   none at all")
            return
    for c in csvs:
        out(f"   {c}  ({os.path.getsize(c)} bytes)")

    target = csvs[0]
    rule(f"SCHEMA — {target}")
    df = pd.read_csv(target)
    out("shape:", df.shape)
    out("\ndtypes:")
    out(df.dtypes.to_string())

    rule("LOW-CARDINALITY COLUMNS (the grouping keys)")
    for col in df.columns:
        try:
            nu = df[col].nunique(dropna=False)
        except TypeError:
            continue
        if nu <= 20:
            vals = df[col].value_counts(dropna=False).to_dict()
            out(f"\n{col}  ({nu} unique)")
            for k, v in vals.items():
                out(f"    {repr(k):30s} n={v}")

    rule("NULL COUNTS PER COLUMN")
    nulls = df.isna().sum()
    out(nulls[nulls > 0].to_string() if (nulls > 0).any() else "   no nulls anywhere")

    rule("FULL CONTENTS" if len(df) <= 120 else "FIRST 120 ROWS")
    out(df.head(120).to_string())

    def cols_matching(*subs):
        return [c for c in df.columns
                if any(s in c.lower() for s in subs)]

    rule("Q1 — ACCURACY COLUMNS, for rebuilding rf_vs_eegnet")
    acc = cols_matching("acc", "score")
    out("candidate accuracy columns:", acc or "NONE FOUND")
    clf = cols_matching("clf", "classifier", "model", "eegnet")
    out("candidate classifier columns:", clf or "NONE FOUND")
    if acc and clf:
        try:
            key = [c for c in cols_matching("group", "phase") if c in df.columns]
            out("\npivot over", key, "x", clf[0])
            out(df.pivot_table(index=key, columns=clf[0],
                               values=acc[0], aggfunc="mean").to_string())
        except Exception as e:
            out("pivot failed:", e)

    rule("Q2 — EEGNET VARIANCE, is SD 0.003 across folds or across seeds?")
    fold = cols_matching("fold", "seed", "split", "rep")
    out("candidate fold/seed columns:", fold or "NONE FOUND")
    if fold and acc:
        for f in fold:
            out(f"\n{f}: {sorted(df[f].dropna().unique())[:20]}")

    rule("Q3 — NAVIGATION, goal completion 100% or 95.5%?")
    nav = cols_matching("goal", "collision", "block", "step", "complet")
    out("candidate navigation columns:", nav or "NONE FOUND")
    for c in nav:
        out(f"\n{c}:")
        out(df[c].describe().to_string())

    rule("Q4 — AMPLIFICATION, 97.9% or 49.5% of trials amplified?")
    amp = cols_matching("amp", "gain", "factor")
    out("candidate amplification columns:", amp or "NONE FOUND")
    for c in amp:
        out(f"\n{c}:")
        try:
            out(df[c].describe().to_string())
        except Exception as e:
            out("   ", e)

    rule("Q5 — TRIAL AND SUBJECT COUNTS")
    cnt = cols_matching("n_trial", "trial", "n_sub", "subject", "n_epoch", "epoch")
    out("candidate count columns:", cnt or "NONE FOUND")
    for c in cnt:
        try:
            out(f"\n{c}: {df[c].describe().to_string()}")
        except Exception:
            out(f"\n{c}: {df[c].nunique()} unique values")

    rule("PIPELINE FILES PRESENT")
    for f in sorted(glob.glob("*.py")):
        out("   ", f)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/audit.txt", "w") as fh:
        fh.write("\n".join(LINES))
    print("\n\nwrote outputs/audit.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Empirical p-value and manuscript sentence from permutation null accuracies."""

import argparse
import os
import numpy as np

OBSERVED = {"stroke": 0.5404, "healthy": 0.6232}


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
    elif ext in (".csv", ".txt", ".tsv"):
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=None, engine="python")
            num = df.select_dtypes("number")
            if num.shape[1] == 0:
                raise ValueError("no numeric column")
            if num.shape[1] > 1:
                print(f"  columns: {list(num.columns)} — using {num.columns[0]!r}")
            a = num.iloc[:, 0].to_numpy()
        except ImportError:
            a = np.loadtxt(path)
    else:
        raise ValueError(f"unsupported: {ext}")
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if a.max() > 1.5:
        print("  values >1 — treating as percentages, dividing by 100")
        a = a / 100.0
    return a


def main(path, observed, group):
    null = load(path)
    n = null.size
    ge = int((null >= observed).sum())
    p = (1 + ge) / (n + 1)
    floor = 1 / (n + 1)

    print(f"\n=== PERMUTATION NULL — {group} ===")
    print(f"  permutations:   {n}")
    print(f"  null mean:      {null.mean():.4f}   (expect ~0.5000)")
    print(f"  null SD:        {null.std(ddof=1):.4f}")
    print(f"  null 95% range: [{np.percentile(null,2.5):.4f}, {np.percentile(null,97.5):.4f}]")
    print(f"  null max:       {null.max():.4f}")
    print(f"  observed:       {observed:.4f}")
    print(f"  perms >= obs:   {ge}")
    print(f"  empirical p:    {p:.4f}   (floor {floor:.4f} at n={n})")

    off = abs(null.mean() - 0.5)
    if off > 0.02:
        print(f"\n  WARNING: null mean off chance by {off:.4f} — check label shuffling "
              f"is within-subject and respects class balance")
    if p <= floor + 1e-12:
        print("\n  p is AT THE FLOOR — report as p < {:.3f}, not as an exact value".format(floor))

    print("\n--- manuscript sentence ---")
    print(f"  A within-subject label-permutation test ({n} permutations) produced a null "
          f"distribution centered at {null.mean():.3f} (SD {null.std(ddof=1):.3f}). The "
          f"observed {group} within-subject accuracy of {observed:.3f} exceeded "
          f"{100*(1 - ge/n):.1f}% of permuted values "
          f"({'p < %.3f' % floor if p <= floor + 1e-12 else 'p = %.3f' % p}).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="null accuracies (.npy/.csv/.txt)")
    ap.add_argument("--group", default="stroke", choices=["stroke", "healthy"])
    ap.add_argument("--observed", type=float, default=None)
    a = ap.parse_args()
    main(a.path, a.observed if a.observed is not None else OBSERVED[a.group], a.group)
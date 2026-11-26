"""realdata_robustness.py - repeated grouped CV, Riemannian leakage, z-score and re-centring on the real corpora.
Usage: python realdata_robustness.py [--repeats 10] [--groups healthy stroke] [--cache ...] [--tag frozen]"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sklearn                                                     # noqa: E402
from sklearn.ensemble import RandomForestClassifier                # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold    # noqa: E402
from sklearn.pipeline import make_pipeline                         # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from pyriemann.estimation import Covariances                       # noqa: E402
from pyriemann.tangentspace import TangentSpace                    # noqa: E402
from pyriemann.utils.mean import mean_riemann                      # noqa: E402
from pyriemann.utils.base import invsqrtm                          # noqa: E402

from classifier_rf_04 import (fit_csp, apply_csp, bandpass_filter,  # noqa: E402
                              scale_invariant_features)

SFREQ, BAND = 160.0, (8.0, 30.0)
N_CSP, N_TREES, N_FOLDS, SEED = 4, 300, 5, 42
EXPECTED = {"healthy": 3076, "stroke": 1758}

if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 6):
    raise SystemExit("needs scikit-learn >= 1.6 for GroupKFold(shuffle=True); "
                     "requirements.txt pins 1.8.0")


# ---------------------------------------------------------------- features
def zscore_per_subject(X, subj):
    """Per-subject, per-channel z-score over all of that subject's trials and samples."""
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(subj):
        m = subj == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True) + 1e-12
        Z[m] = (X[m] - mu) / sd
    return Z


def recenter_per_subject(C, subj):
    """Whiten each subject's covariances by that subject's Riemannian mean."""
    R = np.empty_like(C)
    for s in np.unique(subj):
        m = subj == s
        W = invsqrtm(mean_riemann(C[m]))
        R[m] = W @ C[m] @ W
    return R


def build(Xraw, subj, norm):
    X = zscore_per_subject(Xraw, subj) if norm == "zscore" else np.asarray(Xraw, dtype=np.float64)
    Si = scale_invariant_features(X, SFREQ)
    Si = Si[0] if isinstance(Si, tuple) else Si
    Xb = bandpass_filter(X, SFREQ, *BAND)
    C = Covariances(estimator="oas").fit_transform(Xb)   # per-trial estimate, nothing fitted across trials
    if norm == "recenter":
        C = recenter_per_subject(C, subj)
    return Si, Xb, C


# ---------------------------------------------------------------- scorers
def score_csp_rf(F, y, splits, leak, seed):
    """F = (Si, Xb). leak=True fits CSP on ALL trials before splitting."""
    Si, Xb = F
    f_all = fit_csp(Xb, y, n_components=N_CSP) if leak else None
    accs = []
    for tr, te in splits:
        f = f_all if leak else fit_csp(Xb[tr], y[tr], n_components=N_CSP)
        if not f:
            continue
        Ftr = np.hstack([Si[tr], apply_csp(Xb[tr], f)])
        Fte = np.hstack([Si[te], apply_csp(Xb[te], f)])
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
        accs.append(rf.fit(Ftr, y[tr]).score(Fte, y[te]))
    return accs


def score_riemann(C, y, splits, leak, seed):
    """leak=True fits tangent space (Riemannian mean) + scaler on ALL trials; LR always inside the fold."""
    pre_all = make_pipeline(TangentSpace(metric="riemann"), StandardScaler()).fit(C) if leak else None
    accs = []
    for tr, te in splits:
        if leak:
            T = pre_all.transform(C)
            lr = LogisticRegression(max_iter=2000, random_state=SEED)
            accs.append(lr.fit(T[tr], y[tr]).score(T[te], y[te]))
        else:
            clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=SEED))
            accs.append(clf.fit(C[tr], y[tr]).score(C[te], y[te]))
    return accs


def subset(F, m):
    return (F[0][m], F[1][m]) if isinstance(F, tuple) else F[m]


def within(scorer, F, y, subj, leak, seed):
    per = []
    for s in np.unique(subj):
        m = subj == s
        ys = y[m]
        if len(ys) < N_FOLDS or len(np.unique(ys)) < 2:
            continue
        ns = min(N_FOLDS, int(np.bincount(ys).min()))
        if ns < 2:
            continue
        sp = list(StratifiedKFold(ns, shuffle=True, random_state=seed).split(np.zeros(len(ys)), ys))
        a = scorer(subset(F, m), ys, sp, leak, seed)
        if a:
            per.append(np.mean(a))
    return np.array(per)


def cross_splits(y, subj, scheme, seed):
    z = np.zeros(len(y))
    if scheme == "grouped_fixed":      # exactly the submitted protocol
        return list(GroupKFold(n_splits=N_FOLDS).split(z, y, groups=subj))
    if scheme == "grouped":
        return list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(z, y, groups=subj))
    if scheme == "pooled":
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(z, y))
    raise ValueError(scheme)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("cache", "clean_all_400.npz"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--within-repeats", type=int, default=1,
                    help="within-subject repeats; 1 reproduces the submitted values")
    ap.add_argument("--groups", nargs="+", default=["healthy", "stroke"])
    ap.add_argument("--tag", default="frozen")
    args = ap.parse_args()

    c = np.load(os.path.join(ROOT, args.cache), allow_pickle=True)
    rows = []

    def add(**kw):
        rows.append(kw)
        r = kw
        spread = r.get("sd_subjects") if r["regime"] == "within" else r.get("sd_repeats")
        print(f"  {r['pipeline']:12s} {r['norm']:9s} {r['regime']:6s} {r['protocol']:26s} "
              f"{r['mean']:.4f}  sd {spread:.4f}")

    for g in args.groups:
        Xraw, y, subj = np.asarray(c[g + "_X"]), np.asarray(c[g + "_y"]).astype(int), np.asarray(c[g + "_s"])
        if EXPECTED.get(g) and len(y) != EXPECTED[g]:
            print(f"!! WARNING {g}: expected {EXPECTED[g]} trials, got {len(y)}")
        print(f"\n=== {g}: {len(y)} trials, {len(np.unique(subj))} subjects ({args.cache}) ===")
        t0 = time.time()

        for norm in ("none", "zscore", "recenter"):
            Si, Xb, C = build(Xraw, subj, norm)
            pipes = [("Riemann+LR", score_riemann, C)]
            if norm != "recenter":
                pipes.insert(0, ("CSP+RF", score_csp_rf, (Si, Xb)))

            for name, scorer, F in pipes:
                base = dict(group=g, pipeline=name, norm=norm, cache=args.cache)
                # ---- within-subject (leakage rows only for un-normalised data)
                if norm == "none":
                    for leak, prot in ((False, "inside_fold"), (True, "before_split")):
                        reps = [within(scorer, F, y, subj, leak, SEED + r) for r in range(args.within_repeats)]
                        add(**base, regime="within", protocol=prot,
                            mean=float(np.mean([v.mean() for v in reps])),
                            sd_subjects=float(np.mean([v.std() for v in reps])),
                            sd_repeats=float(np.std([v.mean() for v in reps])),
                            n_subjects=len(reps[0]), n_repeats=args.within_repeats)

                # ---- cross-subject
                protocols = [("grouped", False)] if norm != "none" else [
                    ("grouped_fixed", False), ("grouped", False), ("grouped", True),
                    ("pooled", False), ("pooled", True)]
                for scheme, leak in protocols:
                    n_rep = 1 if scheme == "grouped_fixed" else args.repeats
                    rep_means, fold_sds = [], []
                    for r in range(n_rep):
                        a = scorer(F, y, cross_splits(y, subj, scheme, SEED + r), leak, SEED + r)
                        rep_means.append(np.mean(a))
                        fold_sds.append(np.std(a))
                    add(**base, regime="cross",
                        protocol=f"{scheme}_{'before_split' if leak else 'inside_fold'}",
                        mean=float(np.mean(rep_means)), sd_repeats=float(np.std(rep_means)),
                        sd_folds=float(np.mean(fold_sds)), min_repeat=float(np.min(rep_means)),
                        max_repeat=float(np.max(rep_means)), n_repeats=n_rep)
        print(f"  ({time.time() - t0:.0f} s)")

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out = os.path.join(ROOT, "outputs", f"robustness_{args.tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""crosssubject_significance.py - is cross-subject accuracy above chance? Subject-level tests + label permutation.
Usage: python crosssubject_significance.py [--group stroke] [--perms 200] [--repeats 5] [--pipelines riemann recenter csp]"""
import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp, wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from realdata_robustness import build, cross_splits, SEED, N_CSP, N_TREES   # noqa: E402
from classifier_rf_04 import fit_csp, apply_csp                             # noqa: E402
from sklearn.ensemble import RandomForestClassifier                         # noqa: E402
from sklearn.linear_model import LogisticRegression                         # noqa: E402
from sklearn.pipeline import make_pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                            # noqa: E402
from pyriemann.tangentspace import TangentSpace                             # noqa: E402


def predict_riemann(F, y, tr, te, seed):
    clf = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    return clf.fit(F[tr], y[tr]).predict(F[te])


def predict_csp(F, y, tr, te, seed):
    Si, Xb = F
    f = fit_csp(Xb[tr], y[tr], n_components=N_CSP)          # CSP fitted inside the fold, on (possibly shuffled) labels
    Ftr = np.hstack([Si[tr], apply_csp(Xb[tr], f)])
    Fte = np.hstack([Si[te], apply_csp(Xb[te], f)])
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=1)
    return rf.fit(Ftr, y[tr]).predict(Fte)


def evaluate(predict, F, y, subj, splits_list):
    """Grouped CV over several fold assignments; returns per-subject accuracy and the overall fold-mean accuracy."""
    ids = np.unique(subj)
    acc = {s: [] for s in ids}
    overall = []
    for r, splits in enumerate(splits_list):
        correct = np.zeros(len(y), dtype=bool)
        for tr, te in splits:
            correct[te] = predict(F, y, tr, te, SEED + r) == y[te]
        overall.append(np.mean([correct[te].mean() for _, te in splits]))   # same statistic as realdata_robustness.py
        for s in ids:
            acc[s].append(correct[subj == s].mean())
    return np.array([np.mean(acc[s]) for s in ids]), float(np.mean(overall))


def shuffle_within_subject(y, subj, rng):
    yp = y.copy()
    for s in np.unique(subj):
        m = np.where(subj == s)[0]
        yp[m] = y[rng.permutation(m)]
    return yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="stroke")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5, help="fold assignments per evaluation")
    ap.add_argument("--pipelines", nargs="+", default=["riemann", "recenter", "csp"])
    ap.add_argument("--cache", default=os.path.join("cache", "clean_all_400.npz"))
    args = ap.parse_args()

    c = np.load(os.path.join(ROOT, args.cache), allow_pickle=True)
    g = args.group
    Xraw, y, subj = np.asarray(c[g + "_X"]), np.asarray(c[g + "_y"]).astype(int), np.asarray(c[g + "_s"])
    splits_list = [cross_splits(y, subj, "grouped", SEED + r) for r in range(args.repeats)]
    print(f"{g}: {len(y)} trials, {len(np.unique(subj))} subjects; {args.repeats} fold assignments; "
          f"{args.perms} permutations; labels shuffled within each subject")
    out = os.path.join(ROOT, "outputs", f"crosssubject_significance_{g}.json")
    results = json.load(open(out)) if os.path.exists(out) else {}
    for name in args.pipelines:
        Si, Xb, C = build(Xraw, subj, "recenter" if name == "recenter" else "none")   # no labels used here
        predict, F = (predict_csp, (Si, Xb)) if name == "csp" else (predict_riemann, C)
        t0 = time.time()
        per, obs = evaluate(predict, F, y, subj, splits_list)
        t_one = time.time() - t0
        t = ttest_1samp(per, 0.5)
        w = wilcoxon(per - 0.5)
        print(f"\n[{name}] observed cross-subject accuracy {obs:.4f}  ({t_one:.0f} s per evaluation)")
        print(f"  subject-level: mean {per.mean():.4f}, {np.sum(per > 0.5)}/{len(per)} subjects above 0.5, "
              f"t-test p = {t.pvalue:.4f}, Wilcoxon p = {w.pvalue:.4f}")
        print(f"  permutation: about {t_one * args.perms / max(os.cpu_count() or 1, 1) / 60:.0f} min on "
              f"{os.cpu_count()} cores ...", flush=True)
        null = Parallel(n_jobs=-1)(delayed(evaluate)(
            predict, F, shuffle_within_subject(y, subj, np.random.default_rng(1000 + k)), subj, splits_list)
            for k in range(args.perms))
        null = np.array([n[1] for n in null])
        n_ge = int(np.sum(null >= obs))
        p = (1 + n_ge) / (1 + len(null))
        print(f"  null mean {null.mean():.4f} +/- {null.std():.4f}, 95th percentile {np.percentile(null, 95):.4f}"
              f"  ->  permutation p = {p:.4f}" + ("  (floor: report as p < %.4f)" % p if n_ge == 0 else ""))
        results[name] = dict(observed=obs, subject_mean=float(per.mean()), n_above=int(np.sum(per > 0.5)),
                             n_subjects=len(per), t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue),
                             null_mean=float(null.mean()), null_sd=float(null.std()),
                             null_p95=float(np.percentile(null, 95)), perm_p=float(p), n_null_ge=n_ge,
                             perms=args.perms, repeats=args.repeats)
        json.dump(results, open(out, "w"), indent=1)       # saved after each pipeline
    print("\nwrote", out)


if __name__ == "__main__":
    main()

"""Riemannian tangent-space baseline under the same leakage controls as CSP+RF."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from data_loader_01 import load_all
from preprocessing_02 import preprocess_all
from classifier_rf_04 import bandpass_filter

PTP_UV, MIN_TRIALS = 400.0, 10
SFREQ, BAND = 160.0, (8.0, 30.0)
N_FOLDS, SEED, MIN_WITHIN = 5, 42, 20
EXPECTED = {"healthy": 3076, "stroke": 1758}

def clf():
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED))

def main():
    c = np.load(os.path.join(ROOT, 'cache', 'clean_all_400.npz'), allow_pickle=True)
    data = {g: (c[g + '_X'], c[g + '_y'], c[g + '_s']) for g in ('healthy', 'stroke')}
    rows = []
    for name in ("healthy", "stroke"):
        X, y, subj = data[name]
        X = bandpass_filter(np.asarray(X), SFREQ, *BAND)
        y, subj = np.asarray(y), np.asarray(subj)
        exp = EXPECTED.get(name)
        if exp and len(y) != exp:
            print("  !! WARNING %s: expected %d trials, got %d" % (name, exp, len(y)))
        print("\n=== %s: %d trials, %d subjects ===" % (name, len(y), len(np.unique(subj))))

        accs, kept = [], []
        for s in np.unique(subj):
            m = subj == s
            ys = y[m]
            if len(ys) < N_FOLDS or len(np.unique(ys)) < 2:
                continue
            n_splits = min(N_FOLDS, int(np.bincount(ys).min()))
            if n_splits < 2:
                continue
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
            accs.append(cross_val_score(clf(), X[m], ys, cv=cv).mean())
            kept.append(s)
        w = np.array(accs)
        c = cross_val_score(clf(), X, y, groups=subj, cv=GroupKFold(n_splits=N_FOLDS))

        print("  within-subject : %.4f +/- %.4f  (n=%d subjects)" % (w.mean(), w.std(), len(w)))
        print("  cross-subject  : %.4f +/- %.4f" % (c.mean(), c.std()))
        rows.append(dict(group=name, classifier="Riemann+LR",
                         within_mean=w.mean(), within_std=w.std(), n_subjects=len(w),
                         cross_mean=c.mean(), cross_std=c.std()))
        pd.DataFrame({"subject": kept, "within_acc": w}).to_csv(
            "outputs/riemann_within_%s.csv" % name, index=False)

    pd.DataFrame(rows).to_csv("outputs/riemann_results.csv", index=False)
    print("\nWrote outputs/riemann_results.csv")

if __name__ == "__main__":
    main()

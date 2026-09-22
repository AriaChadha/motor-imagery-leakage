"""Four-protocol leakage experiment on the REAL corpora, using the project's own
CSP (fit_csp/apply_csp) so the numbers are comparable to the reported pipeline."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

from data_loader_01 import load_all
from preprocessing_02 import preprocess_all
from classifier_rf_04 import fit_csp, apply_csp, bandpass_filter, scale_invariant_features

PTP_UV, MIN_TRIALS = 400.0, 10
SFREQ, BAND = 160.0, (8.0, 30.0)
N_CSP, N_TREES, N_FOLDS, SEED = 4, 300, 5, 42
EXPECTED = {"healthy": 3076, "stroke": 1758}

def rf():
    return RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)

def run_folds(X, y, splits, leak, Si):
    """leak=True fits CSP on ALL data before splitting. Si is the label-free
    band-power/RMS block the reported pipeline concatenates to the CSP features."""
    F_all = fit_csp(X, y, n_components=N_CSP) if leak else None
    accs = []
    for tr, te in splits:
        f = F_all if leak else fit_csp(X[tr], y[tr], n_components=N_CSP)
        if not f:
            continue
        F_tr = np.hstack([Si[tr], apply_csp(X[tr], f)])
        F_te = np.hstack([Si[te], apply_csp(X[te], f)])
        accs.append(rf().fit(F_tr, y[tr]).score(F_te, y[te]))
    return float(np.mean(accs)) if accs else float("nan")

def main():
    c = np.load(os.path.join(ROOT, 'cache', 'clean_all_400.npz'), allow_pickle=True)
    data = {g: (c[g + '_X'], c[g + '_y'], c[g + '_s']) for g in ('healthy', 'stroke')}
    rows = []
    for name in ("healthy", "stroke"):
        X, y, subj = data[name]
        Xraw = np.asarray(X)
        X = bandpass_filter(Xraw, SFREQ, *BAND)
        Si = scale_invariant_features(Xraw, SFREQ)
        if isinstance(Si, tuple): Si = Si[0]
        y, subj = np.asarray(y), np.asarray(subj)
        exp = EXPECTED.get(name)
        if exp and len(y) != exp:
            print("  !! WARNING %s: expected %d trials, got %d" % (name, exp, len(y)))
        print("\n" + "=" * 64 + "\n%s: %d trials, %d subjects\n" % (name.upper(), len(y), len(np.unique(subj))) + "=" * 64)

        wi, wb = [], []
        for s in np.unique(subj):
            m = subj == s
            ys = y[m]
            if len(ys) < N_FOLDS or len(np.unique(ys)) < 2:
                continue
            ns = min(N_FOLDS, int(np.bincount(ys).min()))
            if ns < 2:
                continue
            sp = list(StratifiedKFold(ns, shuffle=True, random_state=SEED).split(np.zeros(len(ys)), ys))
            wi.append(run_folds(X[m], ys, sp, False, Si[m]))
            wb.append(run_folds(X[m], ys, sp, True,  Si[m]))
        wi, wb = np.array(wi), np.array(wb)
        print("\nWITHIN-SUBJECT  (n=%d subjects)" % len(wi))
        print("  CSP inside fold   %.4f +/- %.4f   <- reported" % (wi.mean(), wi.std()))
        print("  CSP before split  %.4f +/- %.4f" % (wb.mean(), wb.std()))
        print("  INFLATION         %+.1f pp" % ((wb.mean() - wi.mean()) * 100))

        grp = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups=subj))
        strat = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        cs = {"inside_grouped": run_folds(X, y, grp, False, Si),
              "before_grouped": run_folds(X, y, grp, True, Si),
              "inside_pooled":  run_folds(X, y, strat, False, Si),
              "before_pooled":  run_folds(X, y, strat, True, Si)}
        base = cs["inside_grouped"]
        print("\nCROSS-SUBJECT")
        for k, lab in [("inside_grouped", "CSP inside fold,  grouped   <- controlled"),
                       ("before_grouped", "CSP before split, grouped"),
                       ("inside_pooled",  "CSP inside fold,  pooled"),
                       ("before_pooled",  "CSP before split, pooled")]:
            print("  %-42s %.4f  (%+.1f pp)" % (lab, cs[k], (cs[k] - base) * 100))

        rows.append(dict(group=name, within_inside=wi.mean(), within_inside_sd=wi.std(),
                         within_before=wb.mean(), within_before_sd=wb.std(),
                         within_inflation_pp=(wb.mean() - wi.mean()) * 100, **cs))

    pd.DataFrame(rows).to_csv("outputs/leakage_real.csv", index=False)
    print("\nWrote outputs/leakage_real.csv")

if __name__ == "__main__":
    main()

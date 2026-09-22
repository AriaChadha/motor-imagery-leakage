"""
classifier_rf_04.py — Random Forest, per-group models

Features per epoch: alpha band power, beta band power, RMS per channel, and
CSP log-variance.

Leakage
-------
CSP is supervised: it uses labels to find spatial filters. Fitting it on all
trials and then cross-validating the resulting features lets test-fold labels
influence the filters. On pure noise with random labels, where true accuracy is
0.500, that inflates within-subject accuracy to about 0.52 and cross-subject to
about 0.60 at this dataset's dimensions. Here CSP is refitted inside every fold
on the training split only.

Band power and RMS use no labels, so they are computed once up front. That is
not leakage, and it keeps the fold loop cheap.

Protocols
---------
  within_subject   StratifiedKFold inside each subject, averaged across
                   subjects. What a per-user calibrated system would achieve.
  cross_subject    GroupKFold with subject as the group. Generalization to a
                   new user.

Trials are never pooled across subjects into an ungrouped split.
"""

import os
import pickle
from typing import Dict, Tuple, List

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

ALPHA, BETA = (8, 13), (13, 30)
N_TREES, N_CV, N_CSP = 300, 5, 4
SFREQ = 160.0
GROUPS = ["healthy", "stroke"]


def band_power(X, sfreq, band):
    """Welch PSD averaged over a band. Returns (n_epochs, n_channels)."""
    nperseg = min(X.shape[-1], int(sfreq))
    freqs, psd = welch(X.astype(np.float64), fs=sfreq, nperseg=nperseg, axis=-1)
    return psd[..., (freqs >= band[0]) & (freqs <= band[1])].mean(axis=-1)


def rms_per_channel(X):
    """RMS amplitude per channel per epoch. Amplitude-sensitive."""
    return np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=2))


def bandpass_filter(X, sfreq, low, high):
    nyq = sfreq / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X.astype(np.float64), axis=-1)


def scale_invariant_features(X, sfreq=SFREQ):
    """Alpha power, beta power and RMS. No labels used.

    Returns the matrix and a map from block name to column span, which is what
    makes the ablation possible.
    """
    a = band_power(X, sfreq, ALPHA)
    b = band_power(X, sfreq, BETA)
    r = rms_per_channel(X)
    blocks, start = {}, 0
    for name, blk in (("alpha", a), ("beta", b), ("rms", r)):
        blocks[name] = slice(start, start + blk.shape[1])
        start += blk.shape[1]
    return np.hstack([a, b, r]), blocks


def _mean_cov(Xb):
    """Mean per-epoch covariance, vectorised.

    A loop over np.cov is the bottleneck once CSP runs inside every fold, so
    epochs are centred once and contracted in a single einsum.
    """
    Z = Xb - Xb.mean(axis=-1, keepdims=True)
    return np.einsum("ect,edt->cd", Z, Z) / ((Z.shape[-1] - 1) * len(Z))


def fit_csp(Xb, y, n_components=N_CSP):
    """Fit one-vs-rest CSP filters on the training split only."""
    classes = np.unique(y)
    if len(classes) < 2:
        return []
    n_comp = min(n_components, Xb.shape[1] - 1)
    filters = []
    for cls in classes:
        pos, neg = np.where(y == cls)[0], np.where(y != cls)[0]
        if len(pos) < 2 or len(neg) < 2:
            continue
        cov_p, cov_n = _mean_cov(Xb[pos]), _mean_cov(Xb[neg])
        try:
            evals, evecs = eigh(cov_p, cov_p + cov_n)
        except Exception:
            continue
        half = max(1, n_comp // 2)
        order = np.concatenate([np.argsort(evals)[::-1][:half],
                                np.argsort(evals)[:half]])
        filters.append(evecs[:, order].T)
    return filters


def apply_csp(Xb, filters):
    """Log-variance of the CSP-projected signals."""
    if not filters:
        return np.zeros((len(Xb), 0))
    out = []
    for W in filters:
        P = np.einsum("kc,ect->ekt", W, Xb)
        out.append(np.log(P.var(axis=-1) + 1e-10))
    return np.hstack(out)


def _make_rf(seed):
    return RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                  n_jobs=-1)


def _score_fold(Si, Xb, y, tr, te, seed, keep=None):
    """One fold: refit CSP on train, build features, fit and score."""
    W = fit_csp(Xb[tr], y[tr])
    S_tr = Si[tr] if keep is None else Si[tr][:, keep]
    S_te = Si[te] if keep is None else Si[te][:, keep]
    F_tr = np.hstack([S_tr, apply_csp(Xb[tr], W)])
    F_te = np.hstack([S_te, apply_csp(Xb[te], W)])
    if F_tr.shape[1] == 0:
        return float("nan")
    return float(_make_rf(seed).fit(F_tr, y[tr]).score(F_te, y[te]))


def _within_subject(Si, Xb, y, subj, seed, keep=None):
    """Per-subject cross-validation, averaged across subjects."""
    per = {}
    for sid in np.unique(subj):
        m = np.where(subj == sid)[0]
        ys = y[m]
        if len(ys) < N_CV or len(np.unique(ys)) < 2:
            continue
        n_splits = min(N_CV, int(np.bincount(ys).min()))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        sc = [_score_fold(Si[m], Xb[m], ys, tr, te, seed, keep)
              for tr, te in cv.split(np.zeros(len(ys)), ys)]
        sc = [v for v in sc if np.isfinite(v)]
        if sc:
            per[str(sid)] = float(np.mean(sc))
    vals = np.array(list(per.values()), dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), {}
    return float(vals.mean()), float(vals.std()), per


def _cross_subject(Si, Xb, y, subj, seed, keep=None):
    """GroupKFold by subject: no subject spans train and test."""
    n_subj = len(np.unique(subj))
    if n_subj < 2:
        return float("nan"), float("nan"), []
    cv = GroupKFold(n_splits=min(N_CV, n_subj))
    sc = [_score_fold(Si, Xb, y, tr, te, seed, keep)
          for tr, te in cv.split(np.zeros(len(y)), y, groups=subj)]
    sc = [v for v in sc if np.isfinite(v)]
    if not sc:
        return float("nan"), float("nan"), []
    return float(np.mean(sc)), float(np.std(sc)), sc


def train_evaluate(data, phase="Phase1", out_dir="outputs", random_state=42,
                   sfreq=SFREQ, n_repeats=1):
    """Train and evaluate a separate Random Forest per group.

    data is {group: (X, y, subj)}. n_repeats reruns with successive seeds; that
    captures fold-assignment and classifier seed variance, not sampling
    variance, which the across-subject standard deviation reports.
    """
    os.makedirs(out_dir, exist_ok=True)
    present = {g: v[0].shape[1] for g, v in data.items() if v[0].shape[0] > 0}
    if not present:
        raise RuntimeError("No usable data in any group.")
    n_ch = min(present.values())
    if len(set(present.values())) > 1:
        print(f"  Channel counts differ {present}; truncating to {n_ch}")

    results, models = {}, {}
    for g in GROUPS:
        if g not in data or data[g][0].shape[0] == 0:
            continue
        X, y, subj = data[g]
        X = X[:, :n_ch, :]
        y = LabelEncoder().fit_transform(np.asarray(y))
        subj = np.asarray(subj)
        if len(np.unique(y)) < 2:
            print(f"  RF [{g}] one class only, skipped")
            continue

        Si, blocks = scale_invariant_features(X, sfreq)
        Xb = bandpass_filter(X, sfreq, 8.0, 30.0)

        w_rep, c_rep, folds_all, per_runs = [], [], [], {}
        for r in range(max(1, n_repeats)):
            seed = random_state + r
            wm, _, ps = _within_subject(Si, Xb, y, subj, seed)
            cm, _, folds = _cross_subject(Si, Xb, y, subj, seed)
            w_rep.append(wm)
            c_rep.append(cm)
            folds_all.extend(folds)
            for sid, v in ps.items():
                per_runs.setdefault(sid, []).append(v)

        per_subject = {sid: float(np.mean(v)) for sid, v in per_runs.items()}
        vals = np.array(list(per_subject.values()), dtype=float)
        w_mean = float(vals.mean()) if vals.size else float("nan")
        w_std = float(vals.std()) if vals.size else float("nan")
        c_mean = float(np.nanmean(c_rep))
        c_std = float(np.nanstd(folds_all)) if folds_all else float("nan")

        results[g] = {
            "within_subject": {"mean": w_mean, "std": w_std,
                               "per_subject": per_subject,
                               "repeat_means": [float(v) for v in w_rep],
                               "repeat_std": float(np.nanstd(w_rep))},
            "cross_subject": {"mean": c_mean, "std": c_std,
                              "folds": [float(v) for v in folds_all],
                              "repeat_means": [float(v) for v in c_rep],
                              "repeat_std": float(np.nanstd(c_rep))},
            "rf_mean": w_mean, "rf_std": w_std,
            "rf_trials": [float(v) for v in per_subject.values()],
            "n_trials": int(len(y)),
            "n_subjects": int(len(np.unique(subj))),
            "n_features": int(Si.shape[1] + N_CSP * len(np.unique(y))),
            "n_repeats": int(max(1, n_repeats)),
            "csp_in_fold": True,
        }
        rep = (f"   [{n_repeats} repeats: within sd {np.nanstd(w_rep):.4f}]"
               if n_repeats > 1 else "")
        print(f"  RF [{g}]  within-subject {w_mean:.3f} +/- {w_std:.3f} "
              f"({len(per_subject)} subjects)   "
              f"cross-subject {c_mean:.3f} +/- {c_std:.3f}{rep}")

        W_full = fit_csp(Xb, y)
        F_full = np.hstack([Si, apply_csp(Xb, W_full)])
        models[g] = {"model": _make_rf(random_state).fit(F_full, y),
                     "csp": W_full, "blocks": blocks, "n_ch": n_ch,
                     "sfreq": sfreq}

    with open(os.path.join(out_dir, f"rf_{phase}.pkl"), "wb") as f:
        pickle.dump(models, f)
    return results


def feature_ablation(X, y, subj, sfreq=SFREQ, random_state=42,
                     protocol="within"):
    """Drop each feature block in turn and report the accuracy it accounts for.

    Defaults to within-subject because that is the estimate sitting above
    chance for the impaired group. Ablating from a cross-subject baseline that
    is already at chance reveals nothing: every removal lands inside noise.
    """
    y = LabelEncoder().fit_transform(np.asarray(y))
    subj = np.asarray(subj)
    Si, blocks = scale_invariant_features(X, sfreq)
    Xb = bandpass_filter(X, sfreq, 8.0, 30.0)
    ev = _within_subject if protocol == "within" else _cross_subject

    full, _, _ = ev(Si, Xb, y, subj, random_state)
    print(f"    full feature set ({protocol}-subject): {full:.4f}")
    out = {"full": full}
    for name, sl in blocks.items():
        keep = np.ones(Si.shape[1], dtype=bool)
        keep[sl] = False
        acc, _, _ = ev(Si, Xb, y, subj, random_state, keep=keep)
        out[name] = acc
        print(f"    without {name:<6}: {acc:.4f}  ({100 * (acc - full):+.2f} pp)")

    accs = []
    for sid in np.unique(subj):
        m = np.where(subj == sid)[0]
        ys = y[m]
        if len(ys) < N_CV or len(np.unique(ys)) < 2:
            continue
        n_splits = min(N_CV, int(np.bincount(ys).min()))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        for tr, te in cv.split(np.zeros(len(ys)), ys):
            accs.append(_make_rf(random_state).fit(Si[m][tr], ys[tr])
                        .score(Si[m][te], ys[te]))
    if accs:
        out["csp"] = float(np.mean(accs))
        print(f"    without csp   : {out['csp']:.4f}  "
              f"({100 * (out['csp'] - full):+.2f} pp)")
    return out


def permutation_test(X, y, subj, sfreq=SFREQ, random_state=42, n_perm=200,
                     protocol="within"):
    """Is the observed accuracy above what label shuffling produces?

    Labels are permuted within each subject, preserving trial count and class
    balance per subject and destroying only the association between signal and
    label. The resulting distribution is the exact null for the above-chance
    claim, and it does not assume chance equals 0.5 exactly.
    """
    y = LabelEncoder().fit_transform(np.asarray(y))
    subj = np.asarray(subj)
    Si, _ = scale_invariant_features(X, sfreq)
    Xb = bandpass_filter(X, sfreq, 8.0, 30.0)
    ev = _within_subject if protocol == "within" else _cross_subject

    obs, _, _ = ev(Si, Xb, y, subj, random_state)
    rng = np.random.default_rng(random_state)
    null = []
    for i in range(n_perm):
        yp = y.copy()
        for sid in np.unique(subj):
            m = subj == sid
            yp[m] = rng.permutation(y[m])
        v, _, _ = ev(Si, Xb, yp, subj, random_state)
        if np.isfinite(v):
            null.append(v)
        if (i + 1) % 25 == 0:
            print(f"      {i + 1}/{n_perm} permutations")
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    print(f"    observed {obs:.4f}   null {null.mean():.4f} +/- {null.std():.4f}"
          f"   p = {p:.4g}  ({len(null)} permutations)")
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p": p,
            "n_perm": len(null), "protocol": protocol}

"""csp_patterns.py - Fig. 2: CSP spatial patterns (Haufe A = (C1+C2)W), healthy vs stroke by laterality sign. Descriptive only. Usage: python csp_patterns.py"""
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader_01 import CANONICAL_CHANS                     # noqa: E402
from classifier_rf_04 import bandpass_filter, _mean_cov        # noqa: E402

SFREQ, MU = 160.0, (8.0, 13.0)
mne.set_log_level("ERROR")

# MNE's standard_1020 montage uses mixed case
CASE = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz"}


def laterality(X, y):
    """Copy of erd_accuracy_corr.py: Cohen's d of the C4-vs-C3 mu laterality index between classes."""
    i3, i4 = CANONICAL_CHANS.index("C3"), CANONICAL_CHANS.index("C4")
    f, p = welch(X.astype(np.float64), fs=SFREQ, nperseg=min(X.shape[-1], int(SFREQ)), axis=-1)
    P = np.log(p[..., (f >= MU[0]) & (f <= MU[1])].mean(-1) + 1e-20)
    li = (P[:, i4] - P[:, i3]) / (np.abs(P[:, i4]) + np.abs(P[:, i3]) + 1e-12)
    a, b = li[y == 1], li[y == 0]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def patterns(Xb, y):
    """Spatial patterns of the two extreme CSP components (class 1 high, class 0 high)."""
    C1, C0 = _mean_cov(Xb[y == 1]), _mean_cov(Xb[y == 0])
    evals, W = eigh(C1, C1 + C0)
    A = (C1 + C0) @ W                       # columns = patterns
    out = []
    for k in (np.argmax(evals), np.argmin(evals)):
        a = np.abs(A[:, k])
        out.append(a / a.max())
    return out


def main():
    c = np.load(os.path.join(ROOT, "cache", "clean_all_400.npz"), allow_pickle=True)
    groups = {"Healthy reference": [], "Stroke, correct laterality": [], "Stroke, reversed laterality": []}
    for g in ("healthy", "stroke"):
        X, y, s = np.asarray(c[g + "_X"]), np.asarray(c[g + "_y"]).astype(int), np.asarray(c[g + "_s"])
        assert X.shape[1] == len(CANONICAL_CHANS), "channel count does not match CANONICAL_CHANS"
        Xb = bandpass_filter(X, SFREQ, 8.0, 30.0)
        for sid in np.unique(s):
            m = s == sid
            if min(np.bincount(y[m], minlength=2)) < 3:
                continue
            pats = patterns(Xb[m], y[m])
            if g == "healthy":
                groups["Healthy reference"].append(pats)
            else:
                d = laterality(X[m], y[m])
                if not np.isfinite(d):
                    continue
                key = "Stroke, correct laterality" if d > 0 else "Stroke, reversed laterality"
                groups[key].append(pats)

    names = [CASE.get(ch, ch) for ch in CANONICAL_CHANS]
    info = mne.create_info(names, SFREQ, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    vmax = max(np.array(p)[:, i, :].mean(axis=0).max() for p in groups.values() if len(p) for i in range(2))
    vmax = float(np.ceil(vmax * 10) / 10)          # one shared colour scale, fitted to the data
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8))
    summary = {}
    for j, (label, pats) in enumerate(groups.items()):
        P = np.array(pats)                  # subjects x 2 x channels
        summary[label] = {"n": len(pats)}
        if len(pats) == 0:
            for i in range(2):
                axes[i, j].axis("off")
            axes[0, j].set_title(f"{label}\n(n = 0)", fontsize=10)
            print(f"  !! no subjects in group '{label}'")
            continue
        for i, comp in enumerate(("class 1 component", "class 0 component")):
            mean = P[:, i, :].mean(axis=0)
            summary[label][comp] = {ch: round(float(v), 3) for ch, v in zip(CANONICAL_CHANS, mean)}
            im, _ = mne.viz.plot_topomap(mean, info, axes=axes[i, j], show=False, cmap="Reds",
                                         vlim=(0, vmax), contours=0, sensors=True, extrapolate="local",
                                         names=[n if n in ("C3", "C4") else "" for n in names])
            if i == 0:
                axes[i, j].set_title(f"{label}\n(n = {len(pats)})", fontsize=10)
        axes[0, 0].set_ylabel("Class 1\ncomponent", fontsize=10)
        axes[1, 0].set_ylabel("Class 0\ncomponent", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="|pattern| (normalised)")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = os.path.join(ROOT, "figures", "fig2_csp_patterns.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    json.dump(summary, open(os.path.join(ROOT, "outputs", "csp_patterns.json"), "w"), indent=1)
    print("wrote", out)
    for label, v in summary.items():
        if v["n"] == 0:
            continue
        r, l = v["class 1 component"], v["class 0 component"]
        print(f"  {label:28s} n={v['n']:3d}  class-1 comp C3 {r['C3']:.2f} C4 {r['C4']:.2f}"
              f" | class-0 comp C3 {l['C3']:.2f} C4 {l['C4']:.2f}")


if __name__ == "__main__":
    main()

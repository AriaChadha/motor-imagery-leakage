import os, sys, argparse
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, welch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader_01 import CANONICAL_CHANS

SFREQ, MU = 160.0, (8.0, 13.0)
GROUPS = ["healthy", "stroke"]
COLORS = {"healthy": "#2196F3", "stroke": "#F44336"}

def load(ptp, use_cache):
    c = os.path.join(ROOT, "cache", f"clean_all_{ptp:g}.npz")
    if use_cache and os.path.exists(c):
        print(f"Using {c}")
        d = np.load(c, allow_pickle=True)
        return {g: (d[f"{g}_X"], d[f"{g}_y"], d[f"{g}_s"]) for g in GROUPS}
    from data_loader_01 import load_all
    from preprocessing_02 import preprocess_all
    return preprocess_all(load_all(ROOT), ptp_threshold_uv=ptp)

def mu_power(X):
    f, psd = welch(X.astype(np.float64), fs=SFREQ,
                   nperseg=min(X.shape[-1], int(SFREQ)), axis=-1)
    return psd[..., (f >= MU[0]) & (f <= MU[1])].mean(axis=-1)

def dz(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptp", type=float, default=400.0)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    data = load(a.ptp, not a.no_cache)
    i3, i4 = CANONICAL_CHANS.index("C3"), CANONICAL_CHANS.index("C4")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    S = {}
    for j, g in enumerate(GROUPS):
        X, y, subj = data[g]; y = np.asarray(y).astype(int)
        print("\n" + "="*62)
        print(f"{g.upper()}  {X.shape[0]} trials, {len(np.unique(subj))} "
              f"subjects, classes={np.bincount(y)}")
        print("="*62)
        P = np.log(mu_power(X) + 1e-20)
        d = np.array([dz(P[y==1, c], P[y==0, c]) for c in range(P.shape[1])])
        top = np.argsort(-np.abs(d))[:5]
        print("\nLargest per-channel effects (log mu power, class1 vs class0)")
        for i in top: print(f"  {CANONICAL_CHANS[i]:<5} d = {d[i]:+.3f}")
        print(f"  C3    d = {d[i3]:+.3f}   (expect negative)")
        print(f"  C4    d = {d[i4]:+.3f}   (expect positive)")
        li = (P[:, i4] - P[:, i3]) / (np.abs(P[:, i4]) + np.abs(P[:, i3]) + 1e-12)
        t, p = stats.ttest_ind(li[y==1], li[y==0], equal_var=False)
        dli = dz(li[y==1], li[y==0])
        print(f"\nLaterality index, class1 minus class0")
        print(f"  t = {t:+.3f}   p = {p:.4g}   d = {dli:+.3f}")
        b, aa = butter(4, [MU[0]/(SFREQ/2), MU[1]/(SFREQ/2)], btype="band")
        F = filtfilt(b, aa, X[:, [i3, i4], :].astype(np.float64), axis=-1)
        nb = int(0.5*SFREQ); nbins = X.shape[-1]//nb
        tr = []
        for k in range(nbins):
            pw = np.log((F[..., k*nb:(k+1)*nb]**2).mean(-1) + 1e-20)
            df = pw[:, 0] - pw[:, 1]
            tr.append(dz(df[y==1], df[y==0]))
        tr = np.array(tr)
        print("\nDiscriminability across epoch, 0.5 s bins")
        print("  " + "  ".join(f"{v:+.2f}" for v in tr))
        S[g] = dict(d3=d[i3], d4=d[i4], dli=dli, p=p)
        cols = [COLORS[g] if c in ("C3","C4") else "#444" for c in CANONICAL_CHANS]
        ax[0, j].bar(range(len(d)), d, color=cols, alpha=.85)
        ax[0, j].axhline(0, color="k", lw=.8)
        ax[0, j].set_xticks(range(len(d)))
        ax[0, j].set_xticklabels(CANONICAL_CHANS, rotation=90, fontsize=6)
        ax[0, j].set_ylabel("Cohen's d, log mu power")
        ax[0, j].set_title(f"{g.capitalize()}: class separation per channel",
                           fontweight="bold", fontsize=11)
        ax[0, j].grid(axis="y", alpha=.3)
        ax[1, j].plot(np.arange(nbins)*.5 + .25, tr, "o-", color=COLORS[g], lw=2)
        ax[1, j].axhline(0, color="k", lw=.8)
        ax[1, j].set_xlabel("Time within epoch (s)")
        ax[1, j].set_ylabel("Cohen's d, C3 minus C4")
        ax[1, j].set_title(f"{g.capitalize()}: lateralization over epoch",
                           fontweight="bold", fontsize=11)
        ax[1, j].grid(alpha=.3)
    od = os.path.join(ROOT, "outputs", "figures"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "erd_check.png")
    plt.tight_layout(); plt.savefig(fp, dpi=150); plt.close()
    h, s = S["healthy"], S["stroke"]
    hok = (h["d3"] < 0 and h["d4"] > 0) or (abs(h["dli"]) > .15 and h["p"] < .01)
    sok = (s["d3"] < 0 and s["d4"] > 0) and abs(s["dli"]) > .15 and s["p"] < .01
    print("\n" + "="*62 + "\nREADING\n" + "="*62)
    if not hok:
        print("The healthy positive control shows no contralateral ERD either.\n"
              "That points at something shared by both groups: channel mapping,\n"
              "epoch window, or band. Fix the healthy path before concluding\n"
              "anything about stroke.")
    elif sok:
        print("Contralateral ERD is present in the stroke epochs with the\n"
              "expected sign. Epochs are aligned and labels are correct, so\n"
              "chance-level cross-subject accuracy is a genuine finding about\n"
              "transfer, not a preprocessing failure.")
    elif abs(s["dli"]) < .15 or s["p"] > .05:
        print("Healthy shows lateralized ERD; stroke shows essentially none.\n"
              "Same pipeline, same channels, same band, so this points to the\n"
              "stroke epochs rather than to physiology. Next: the marker window\n"
              "in _load_stroke_mat, whether MI_CODE = 2 marks imagery onset and\n"
              "whether the 4 s from it lands on imagery rather than instruction\n"
              "or rest.")
    else:
        print("Stroke shows a weak or wrong-signed effect: not clean enough to\n"
              "call the null genuine, not absent enough to call it a bug. Check\n"
              "the time course: a peak away from mid-epoch suggests a window\n"
              "offset, a flat line suggests the window misses imagery entirely.")
    print(f"\nFigure: {fp}")

if __name__ == "__main__":
    main()

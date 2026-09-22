import os, sys
import numpy as np
from scipy import stats
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader_01 import CANONICAL_CHANS

SFREQ, MU = 160.0, (8.0, 13.0)
GROUPS = ["healthy", "stroke"]
COLORS = {"healthy": "#2196F3", "stroke": "#F44336"}

d = np.load(os.path.join(ROOT, "cache", "clean_all_400.npz"), allow_pickle=True)
data = {g: (d[f"{g}_X"], d[f"{g}_y"], d[f"{g}_s"]) for g in GROUPS}
i3, i4 = CANONICAL_CHANS.index("C3"), CANONICAL_CHANS.index("C4")

def mu_pow(X):
    f, p = welch(X.astype(np.float64), fs=SFREQ,
                 nperseg=min(X.shape[-1], int(SFREQ)), axis=-1)
    return p[..., (f >= MU[0]) & (f <= MU[1])].mean(-1)

def dz(a, b):
    if len(a) < 3 or len(b) < 3: return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(sp+1e-12))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for j, g in enumerate(GROUPS):
    X, y, s = data[g]; y = np.asarray(y).astype(int); s = np.asarray(s)
    per = {}
    for sid in np.unique(s):
        m = s == sid
        if len(np.unique(y[m])) < 2: continue
        P = np.log(mu_pow(X[m]) + 1e-20)
        li = (P[:, i4]-P[:, i3])/(np.abs(P[:, i4])+np.abs(P[:, i3])+1e-12)
        v = dz(li[y[m] == 1], li[y[m] == 0])
        if np.isfinite(v): per[sid] = v
    v = np.array(list(per.values()))
    pos = float((v > 0).mean())
    sign_p = stats.binomtest(int((v > 0).sum()), len(v), 0.5).pvalue
    print(f"\n{g.upper()}  {len(v)} subjects")
    print(f"  mean d           {v.mean():+.3f}")
    print(f"  mean |d|         {np.abs(v).mean():+.3f}")
    print(f"  cancellation     {np.abs(v).mean()/ (abs(v.mean())+1e-9):.1f}x")
    print(f"  correct sign     {100*pos:.0f}%  (sign test p={sign_p:.4g})")
    print(f"  range            {v.min():+.3f} to {v.max():+.3f}")
    strong = np.abs(v) > 0.3
    print(f"  |d|>0.3          {strong.sum()}/{len(v)} subjects "
          f"({100*(v[strong] > 0).mean() if strong.sum() else 0:.0f}% positive)")
    ax[j].hist(v, bins=18, color=COLORS[g], alpha=.85, edgecolor="white")
    ax[j].axvline(0, color="k", lw=1.2)
    ax[j].axvline(v.mean(), color="darkgreen", ls="--", lw=2, label=f"mean {v.mean():+.3f}")
    ax[j].set_xlabel("Per-subject laterality effect (Cohen's d)")
    ax[j].set_ylabel("Subjects")
    ax[j].set_title(f"{g.capitalize()}: {len(v)} subjects", fontweight="bold")
    ax[j].legend(fontsize=9); ax[j].grid(axis="y", alpha=.3)
od = os.path.join(ROOT, "outputs", "figures"); os.makedirs(od, exist_ok=True)
fp = os.path.join(od, "subject_laterality.png")
plt.tight_layout(); plt.savefig(fp, dpi=150); plt.close()
print(f"\nFigure: {fp}")

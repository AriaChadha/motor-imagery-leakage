import os
import sys
import json
import numpy as np
from scipy import stats
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader_01 import CANONICAL_CHANS

SFREQ, MU = 160.0, (8.0, 13.0)
d = np.load(os.path.join(ROOT, "cache", "clean_all_400.npz"), allow_pickle=True)
X, y, s = d["stroke_X"], np.asarray(d["stroke_y"]).astype(int), np.asarray(d["stroke_s"])
i3, i4 = CANONICAL_CHANS.index("C3"), CANONICAL_CHANS.index("C4")


def mu_pow(A):
    f, p = welch(A.astype(np.float64), fs=SFREQ,
                 nperseg=min(A.shape[-1], int(SFREQ)), axis=-1)
    return p[..., (f >= MU[0]) & (f <= MU[1])].mean(-1)


def dz(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


lat = {}
for sid in np.unique(s):
    m = s == sid
    if len(np.unique(y[m])) < 2:
        continue
    P = np.log(mu_pow(X[m]) + 1e-20)
    li = (P[:, i4] - P[:, i3]) / (np.abs(P[:, i4]) + np.abs(P[:, i3]) + 1e-12)
    v = dz(li[y[m] == 1], li[y[m] == 0])
    if np.isfinite(v):
        lat[str(sid)] = v

res = json.load(open(os.path.join(ROOT, "outputs", "pipeline_results.json")))
acc = res["phase_results"]["Phase1"]["stroke"]["within_subject"]["per_subject"]

ids = sorted(set(lat) & set(acc))
print(f"laterality {len(lat)}  accuracy {len(acc)}  matched {len(ids)}")
if len(ids) < 10:
    print("KEY MISMATCH")
    print("  lat:", sorted(lat)[:5])
    print("  acc:", sorted(acc)[:5])
    raise SystemExit(1)

L = np.array([lat[i] for i in ids])
A = np.array([acc[i] for i in ids])

rho, p_s = stats.spearmanr(L, A)
r, p_p = stats.pearsonr(L, A)
print(f"\nn = {len(ids)}")
print(f"Spearman rho = {rho:+.3f}  p = {p_s:.4f}")
print(f"Pearson  r   = {r:+.3f}  p = {p_p:.4f}")

hi, lo = A[L > 0], A[L <= 0]
print(f"\npositive laterality  n={len(hi):2d}  acc {hi.mean():.4f} +/- {hi.std():.4f}")
print(f"negative laterality  n={len(lo):2d}  acc {lo.mean():.4f} +/- {lo.std():.4f}")
if len(hi) > 2 and len(lo) > 2:
    t, pt = stats.ttest_ind(hi, lo, equal_var=False)
    print(f"Welch t = {t:+.3f}  p = {pt:.4f}")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(L, A, s=34, color="#F44336", edgecolor="black", linewidth=0.5, zorder=3)
ax.axhline(0.5, color="black", ls=":", lw=1.2)
ax.axvline(0.0, color="grey", ls=":", lw=1.0)
if len(ids) > 2:
    b, a0 = np.polyfit(L, A, 1)
    xs = np.linspace(L.min(), L.max(), 100)
    ax.plot(xs, b * xs + a0, color="#1f77b4", lw=1.6)
ax.set_xlabel("Per-subject laterality effect (Cohen's d)")
ax.set_ylabel("Within-subject accuracy")
ax.set_title(f"Stroke: laterality vs decodability\nSpearman rho = {rho:+.3f}, p = {p_s:.3f}",
             fontsize=10)
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "figures", "erd_vs_accuracy.png")
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")

json.dump({"laterality": lat, "accuracy": {i: acc[i] for i in ids},
           "spearman_rho": rho, "spearman_p": p_s, "n": len(ids)},
          open(os.path.join(ROOT, "outputs", "erd_accuracy.json"), "w"), indent=2)
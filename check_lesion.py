import os, sys, csv, re
import numpy as np
from scipy import stats
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader_01 import CANONICAL_CHANS

SFREQ, MU = 160.0, (8.0, 13.0)
N_PERM = 500
TSV = os.path.join(ROOT, "data", "figshare_stroke", "participants.tsv")
CORTICAL = ("cortex","cortical","frontal","parietal","temporal","occipital",
            "insula","corona","centrum")
SUBCORTICAL = ("pons","medulla","midbrain","cerebell","thalam","basal",
               "capsule","putamen","lacunar","brainstem")

def mu_pow(X):
    f, p = welch(X.astype(np.float64), fs=SFREQ,
                 nperseg=min(X.shape[-1], int(SFREQ)), axis=-1)
    return p[..., (f >= MU[0]) & (f <= MU[1])].mean(-1)

def dz(a, b):
    if len(a) < 3 or len(b) < 3: return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))
                 / (len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(sp+1e-12))

def read_meta():
    if not os.path.exists(TSV): return {}
    out = {}
    with open(TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m = re.search(r"(\d+)", row.get("Participant_ID",""))
            if not m: continue
            loc = (row.get("StrokeLocation") or "").lower()
            kind = ("cortical" if any(k in loc for k in CORTICAL)
                    else "subcortical" if any(k in loc for k in SUBCORTICAL)
                    else "other")
            out[f"stroke_{int(m.group(1)):02d}"] = {
                "paralysis": (row.get("ParalysisSide") or "").strip().lower(),
                "location": row.get("StrokeLocation",""), "kind": kind}
    return out

def subject_effects(X, y, s, i3, i4):
    per = {}
    for sid in np.unique(s):
        m = s == sid
        if len(np.unique(y[m])) < 2: continue
        P = np.log(mu_pow(X[m]) + 1e-20)
        li = (P[:,i4]-P[:,i3])/(np.abs(P[:,i4])+np.abs(P[:,i3])+1e-12)
        v = dz(li[y[m]==1], li[y[m]==0])
        if np.isfinite(v): per[sid] = v
    return per

def permutation_null(X, y, s, i3, i4, rng):
    P = np.log(mu_pow(X) + 1e-20)
    li = (P[:,i4]-P[:,i3])/(np.abs(P[:,i4])+np.abs(P[:,i3])+1e-12)
    idx = {sid: np.where(s == sid)[0] for sid in np.unique(s)}
    an, mn = [], []
    for _ in range(N_PERM):
        vals = []
        for sid, ii in idx.items():
            yy = rng.permutation(y[ii])
            if len(np.unique(yy)) < 2: continue
            v = dz(li[ii][yy==1], li[ii][yy==0])
            if np.isfinite(v): vals.append(v)
        vals = np.array(vals)
        an.append(np.abs(vals).mean()); mn.append(vals.mean())
    return np.array(an), np.array(mn)

def describe(name, v):
    v = np.asarray(v)
    if len(v) < 3:
        print(f"  {name:<26} n={len(v)}, too few to test"); return
    t, p = stats.ttest_1samp(v, 0.0)
    pos = int((v > 0).sum())
    sp = stats.binomtest(pos, len(v), 0.5).pvalue
    print(f"  {name:<26} n={len(v):<3} mean d={v.mean():+.3f}  "
          f"|d|={np.abs(v).mean():.3f}  t={t:+.2f} p={p:.3f}  "
          f"{100*pos/len(v):.0f}% positive (p={sp:.3f})")

def main():
    d = np.load(os.path.join(ROOT,"cache","clean_all_400.npz"), allow_pickle=True)
    i3, i4 = CANONICAL_CHANS.index("C3"), CANONICAL_CHANS.index("C4")
    rng = np.random.default_rng(42)
    meta = read_meta()
    print(f"participants.tsv: {len(meta)} subjects" if meta else "participants.tsv missing")
    per_group = {}
    for g in ["healthy","stroke"]:
        X = d[f"{g}_X"]; y = np.asarray(d[f"{g}_y"]).astype(int)
        s = np.asarray(d[f"{g}_s"])
        per = subject_effects(X, y, s, i3, i4); per_group[g] = per
        v = np.array(list(per.values()))
        n_tr = int(np.median([np.sum(s == k) for k in per]))
        print("\n" + "="*70)
        print(f"{g.upper()}  {len(per)} subjects, median {n_tr} trials each")
        print("="*70)
        an, mn = permutation_null(X, y, s, i3, i4, rng)
        oa, om = np.abs(v).mean(), v.mean()
        pa = float((an >= oa).mean()); pm = float((np.abs(mn) >= abs(om)).mean())
        print(f"  observed mean |d| {oa:.3f}   null {an.mean():.3f} +/- {an.std():.3f}   p = {pa:.4f}")
        print(f"  observed mean  d  {om:+.3f}   null {mn.mean():+.3f} +/- {mn.std():.3f}   p = {pm:.4f}")
        print("  -> individual subjects carry real lateralization beyond noise" if pa < .05
              else "  -> mean |d| is within the noise floor; individual effects not established")
    if not meta: return
    per = per_group["stroke"]
    print("\n" + "="*70); print("STROKE SPLIT BY LESION"); print("="*70)
    for key, label in (("paralysis","Paralysis side"), ("kind","Lesion type")):
        print(f"\n{label}:")
        groups = {}
        for sid, v in per.items():
            k = meta.get(sid, {}).get(key)
            if k: groups.setdefault(k, []).append(v)
        for k in sorted(groups): describe(k, groups[k])
        if len(groups) == 2:
            a, b = [np.array(groups[k]) for k in sorted(groups)]
            if len(a) >= 3 and len(b) >= 3:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                print(f"  {'difference between them':<26} t={t:+.2f}  p={p:.4f}  d={dz(a,b):+.3f}")
                if p < .05:
                    print(f"  -> lateralization depends on {label.lower()}; the group\n"
                          f"     average cancels because subgroups point different ways")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(list(per_group["healthy"].values()), bins=18, color="#2196F3",
               alpha=.85, edgecolor="white")
    ax[0].set_title("Healthy", fontweight="bold")
    groups = {}
    for sid, v in per.items():
        k = meta.get(sid, {}).get("paralysis")
        if k: groups.setdefault(k, []).append(v)
    for k, c in zip(sorted(groups), ["#F44336","#FF9800"]):
        ax[1].hist(groups[k], bins=12, alpha=.6, color=c, edgecolor="white",
                   label=f"{k} paralysis (n={len(groups[k])})")
    ax[1].legend(fontsize=9); ax[1].set_title("Stroke by lesion side", fontweight="bold")
    for a in ax:
        a.axvline(0, color="k", lw=1.2); a.grid(axis="y", alpha=.3)
        a.set_xlabel("Per-subject laterality (Cohen's d)"); a.set_ylabel("Subjects")
    od = os.path.join(ROOT,"outputs","figures"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od,"lesion_laterality.png")
    plt.tight_layout(); plt.savefig(fp, dpi=150); plt.close()
    print(f"\nFigure: {fp}")

if __name__ == "__main__":
    main()

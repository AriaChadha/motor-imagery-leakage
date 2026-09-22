"""
check_leakage.py — quantify evaluation leakage on label-free data.

Generates synthetic EEG in which the labels carry NO information about the
signal, then runs the classification pipeline under controlled and uncontrolled
protocols. Any accuracy above 0.50 is manufactured by the protocol.

Two leakage sources are separated:
  1. Supervised filter fitting location — CSP fitted before the split sees test
     labels during feature construction.
  2. Fold grouping — pooled stratified folds place the same subject on both
     sides of a split, so the model can exploit per-subject class imbalance.

Source 2 requires two realistic properties of the synthetic data, both of which
hold in real corpora and both of which are stated explicitly here:
  - subjects have distinct spatial covariance, so subject identity is learnable
  - per-subject class proportions are not exactly balanced

Outputs figures/leakage_quantification.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold

mne.set_log_level("ERROR")

# ----------------------------------------------------------------------
# configuration — match these to the real pipeline where it matters
# ----------------------------------------------------------------------
N_SUBJECTS = 46
N_TRIALS = 38          # 1758 stroke trials / 46 subjects
N_CHANNELS = 22
N_TIMES = 96
N_COMPONENTS = 4       # CSP components
N_TREES = 80           # keep low; this is a null experiment, not a benchmark
N_FOLDS = 5
N_REPEATS = 20
SEED = 0

OUTDIR = "figures"
OUTFILE = os.path.join(OUTDIR, "leakage_quantification.png")


def make_null_data(rng, balanced):
    """Label-free data. y is independent of X by construction.

    balanced=True  -> exactly 50/50 per subject, so chance is exactly 0.50 and
                      any excess accuracy is manufactured by the protocol.
    balanced=False -> per-subject class proportions vary, which is what makes
                      subject identity exploitable under pooled stratified folds.
                      The reference is then the majority-class rate, not 0.50.
    """
    X_all, y_all, g_all = [], [], []
    for s in range(N_SUBJECTS):
        # subject-specific spatial mixing so subject identity is learnable
        mixing = rng.normal(size=(N_CHANNELS, N_CHANNELS)) / np.sqrt(N_CHANNELS)

        if balanced:
            y = np.zeros(N_TRIALS, dtype=int)
            y[: N_TRIALS // 2] = 1
            rng.shuffle(y)
        else:
            p = rng.beta(8.0, 8.0)
            y = (rng.random(N_TRIALS) < p).astype(int)
            if y.sum() < 2:
                y[:2] = 1
            if (1 - y).sum() < 2:
                y[-2:] = 0

        noise = rng.normal(size=(N_TRIALS, N_CHANNELS, N_TIMES))
        X = np.einsum("ij,tjk->tik", mixing, noise)

        X_all.append(X)
        y_all.append(y)
        g_all.append(np.full(N_TRIALS, s))

    return (np.concatenate(X_all).astype(np.float64),
            np.concatenate(y_all),
            np.concatenate(g_all))


def _clf(seed):
    return RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
                                  random_state=seed)


def score_folds(X, y, splits, csp_inside, seed):
    """Run one protocol. csp_inside=False fits CSP on all data before splitting."""
    if not csp_inside:
        csp_all = CSP(n_components=N_COMPONENTS, log=True, norm_trace=False)
        F_all = csp_all.fit_transform(X, y)          # <-- leak: sees every label

    accs = []
    for tr, te in splits:
        if csp_inside:
            csp = CSP(n_components=N_COMPONENTS, log=True, norm_trace=False)
            F_tr = csp.fit_transform(X[tr], y[tr])   # training labels only
            F_te = csp.transform(X[te])
        else:
            F_tr, F_te = F_all[tr], F_all[te]

        clf = _clf(seed)
        clf.fit(F_tr, y[tr])
        accs.append(clf.score(F_te, y[te]))
    return float(np.mean(accs))


def run():
    rng_master = np.random.default_rng(SEED)
    results = {k: [] for k in
               ["within_out", "within_in",
                "cross_out_strat", "cross_out_group",
                "cross_in_strat", "cross_in_group", "baseline"]}

    for rep in range(N_REPEATS):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        seed = int(rng.integers(1 << 31))

        # ---- within-subject: balanced labels, so chance is exactly 0.50 ----
        X, y, g = make_null_data(rng, balanced=True)

        w_out, w_in = [], []
        for s in range(N_SUBJECTS):
            m = g == s
            Xs, ys = X[m], y[m]
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            splits = list(skf.split(Xs, ys))
            w_out.append(score_folds(Xs, ys, splits, csp_inside=False, seed=seed))
            w_in.append(score_folds(Xs, ys, splits, csp_inside=True, seed=seed))
        results["within_out"].append(np.mean(w_out))
        results["within_in"].append(np.mean(w_in))

        # ---- cross-subject: per-subject imbalance, which is what makes fold
        # ---- grouping matter. Reference is the majority-class rate.
        X, y, g = make_null_data(rng, balanced=False)
        results["baseline"].append(np.mean([
            max(y[g == s].mean(), 1 - y[g == s].mean()) for s in range(N_SUBJECTS)]))

        strat = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                     random_state=seed).split(X, y))
        group = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups=g))

        results["cross_out_strat"].append(score_folds(X, y, strat, False, seed))
        results["cross_out_group"].append(score_folds(X, y, group, False, seed))
        results["cross_in_strat"].append(score_folds(X, y, strat, True, seed))
        results["cross_in_group"].append(score_folds(X, y, group, True, seed))

        print(f"  repeat {rep + 1}/{N_REPEATS} done")

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def plot(res):
    os.makedirs(OUTDIR, exist_ok=True)
    UNCTRL, CTRL = "#e15759", "#4e79a7"
    base = res["baseline"][0]
    ref = res["cross_in_group"][0]          # correct protocol = the reference

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.9]})

    # ---- panel A: within-subject, balanced labels, chance = 0.50 ----
    ax = axes[0]
    keys = ["within_out", "within_in"]
    labels = ["CSP fitted\nbefore split", "CSP fitted\ninside fold"]
    means = [res[k][0] for k in keys]
    errs = [res[k][1] for k in keys]
    bars = ax.bar(labels, means, yerr=errs, capsize=4,
                  color=[UNCTRL, CTRL], edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.3, zorder=0)
    ax.text(0.015, 0.5, "chance 0.50", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=8)
    ax.annotate(f"+{(means[0] - 0.5) * 100:.1f} pp", xy=(0, means[0]),
                xytext=(0, means[0] + 0.045), ha="center",
                fontsize=10, fontweight="bold", color=UNCTRL)
    ax.set_title("Within-subject regime\nbalanced labels",
                 fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Accuracy on label-free data")

    # ---- panel B: cross-subject, referenced to the correct protocol ----
    ax2 = axes[1]
    keys2 = ["cross_out_strat", "cross_out_group", "cross_in_strat", "cross_in_group"]
    labels2 = ["CSP before split\npooled stratified",
               "CSP before split\ngrouped by subject",
               "CSP inside fold\npooled stratified",
               "CSP inside fold\ngrouped by subject"]
    means2 = [res[k][0] for k in keys2]
    errs2 = [res[k][1] for k in keys2]
    bars2 = ax2.bar(labels2, means2, yerr=errs2, capsize=4,
                    color=[UNCTRL, UNCTRL, UNCTRL, CTRL],
                    edgecolor="black", linewidth=0.6)
    ax2.axhline(ref, color=CTRL, linestyle="-", linewidth=1.4, zorder=0)
    ax2.text(0.015, ref, f"correct protocol {ref:.3f}",
             transform=ax2.get_yaxis_transform(), va="bottom",
             fontsize=8, color=CTRL)
    ax2.axhline(base, color="darkgreen", linestyle="--", linewidth=1.2, zorder=0)
    ax2.text(0.985, base, f"majority-class rate {base:.3f}",
             transform=ax2.get_yaxis_transform(), va="bottom", ha="right",
             fontsize=8, color="darkgreen")
    for b, m in zip(bars2, means2):
        if abs(m - ref) > 1e-9:
            ax2.text(b.get_x() + b.get_width() / 2, m + 0.014,
                     f"+{(m - ref) * 100:.1f} pp", ha="center",
                     fontsize=9.5, fontweight="bold", color=UNCTRL)
    ax2.set_title("Cross-subject regime\ninflation relative to the correct protocol",
                  fontweight="bold", fontsize=10.5)

    for a, ms in ((ax, means), (ax2, means2)):
        a.set_ylim(0.44, max(ms) + 0.10)
        a.tick_params(axis="x", labelsize=8.5)
        a.grid(axis="y", alpha=0.25)
        a.set_axisbelow(True)
        for b, m in zip(a.patches, ms):
            a.text(b.get_x() + b.get_width() / 2, m - 0.022,
                   f"{m:.3f}", ha="center", fontsize=8.5, color="white",
                   fontweight="bold")

    fig.suptitle("Accuracy manufactured by evaluation protocol on data containing no class information",
                 fontweight="bold", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTFILE, dpi=150)
    print(f"\nwrote {OUTFILE}")

if __name__ == "__main__":
    print("running null experiment...")
    res = run()
    print()
    for k, (m, s) in res.items():
        print(f"  {k:20s} {m:.4f} +/- {s:.4f}   ({(m - 0.5) * 100:+.2f} pp)")
    plot(res)
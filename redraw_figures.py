"""redraw_figures.py - publication versions of Figs 5, 8, 10 and 11, drawn only from saved
outputs (outputs/pipeline_results.json, outputs/statistics.json, outputs/robustness_frozen.csv). Nothing is recomputed.
Usage: python redraw_figures.py        -> figures/fig5_..., fig8_..., fig10_..., fig11_...png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"healthy": "#4e79a7", "stroke": "#e15759"}
NAME = {"healthy": "Healthy reference", "stroke": "Stroke"}
PH = ["Phase1", "Phase2", "Phase3"]
ALPHA = {"Phase1": 1.0, "Phase2": 0.65, "Phase3": 0.35}

r = json.load(open(os.path.join(ROOT, "outputs", "pipeline_results.json")))
stats = json.load(open(os.path.join(ROOT, "outputs", "statistics.json")))
P = r["phase_results"]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def chance(ax):
    ax.axhline(0.5, color="black", ls=":", lw=1, zorder=0)


# ---- Fig. 5: per-user calibration vs transfer to a new user (repeated grouped CV, as in Table 4)
rb = pd.read_csv(os.path.join(ROOT, "outputs", "robustness_frozen.csv"))
sel = rb[(rb.pipeline == "CSP+RF") & (rb.norm == "none")]
fig, ax = plt.subplots(figsize=(3.4, 2.8))
x = np.arange(2)
for j, (reg, prot, spread, lab, a) in enumerate([
        ("within", "inside_fold", "sd_subjects", "Within-subject (calibrated)", 1.0),
        ("cross", "grouped_inside_fold", "sd_repeats", "Cross-subject (new user)", 0.45)]):
    m = [sel[(sel.group == g) & (sel.regime == reg) & (sel.protocol == prot)]["mean"].iloc[0] for g in COL]
    e = [sel[(sel.group == g) & (sel.regime == reg) & (sel.protocol == prot)][spread].iloc[0] for g in COL]
    ax.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=e, capsize=3, color=list(COL.values()), alpha=a,
           edgecolor="black", linewidth=0.5, label=lab)
    for xi, v in zip(x + (j - 0.5) * 0.38, m):
        ax.text(xi, 0.405, f"{v:.3f}", ha="center", fontsize=6.5, color="white" if a == 1 else "black")
chance(ax)
ax.set_xticks(x)
ax.set_xticklabels([NAME[g] for g in COL])
ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 0.8)
h = [plt.Rectangle((0, 0), 1, 1, color="grey", alpha=a, ec="black", lw=0.5) for a in (1.0, 0.45)]
ax.legend(h, ["Within-subject (calibrated)", "Cross-subject (new user)"], loc="upper right", frameon=False)
save(fig, "fig5_calibration_vs_transfer.png")

# ---- Fig. 8: within-subject accuracy by group and phase
fig, ax = plt.subplots(figsize=(3.6, 2.8))
w = 0.26
for i, ph in enumerate(PH):
    m = [P[ph][g]["rf_mean"] for g in COL]
    e = [P[ph][g]["rf_std"] for g in COL]
    ax.bar(np.arange(2) + (i - 1) * w, m, w * 0.94, yerr=e, capsize=2.5, color=list(COL.values()),
           alpha=ALPHA[ph], edgecolor="black", linewidth=0.5)
chance(ax)
ax.set_xticks(np.arange(2))
ax.set_xticklabels([NAME[g] for g in COL])
ax.set_ylabel("Within-subject accuracy")
ax.set_ylim(0.3, 0.85)
h = [plt.Rectangle((0, 0), 1, 1, color="grey", alpha=ALPHA[p], ec="black", lw=0.5) for p in PH]
ax.legend(h, ["Phase 1, no gain", "Phase 2, population reference", "Phase 3, subject reference"],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
save(fig, "fig8_accuracy_by_phase.png")

# ---- Fig. 10: change in stroke accuracy under each amplification design, with 95% CI
fig, ax = plt.subplots(figsize=(3.0, 2.6))
rows = [stats[f"{ph}|stroke"] for ph in ("Phase2", "Phase3")]
m = [s["mean_change_pp"] for s in rows]
lo = [s["mean_change_pp"] - s["ci_low_pp"] for s in rows]
hi = [s["ci_high_pp"] - s["mean_change_pp"] for s in rows]
ax.bar([0, 1], m, 0.55, yerr=[lo, hi], capsize=4, color=COL["stroke"], alpha=0.8, edgecolor="black", linewidth=0.5)
for xi, s in zip([0, 1], rows):
    ax.text(xi, s["ci_high_pp"] + 0.25, f"{s['mean_change_pp']:+.2f} pp\np = {s['p_paired_t']:.3f}",
            ha="center", fontsize=7)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Phase 2\npopulation reference", "Phase 3\nsubject reference"])
ax.set_ylabel("Change in stroke accuracy (pp)")
ax.set_ylim(-4, 4.2)
ax.set_xlim(-0.6, 1.6)
save(fig, "fig10_amplification_effect.png")

# ---- Fig. 11: navigation outcomes
nav = r["navigation"]
metrics = [("Goal completion (%)", lambda rs: 100 * np.mean([t["reached_goal"] for t in rs])),
           ("Steps to goal", lambda rs: np.mean([t["n_steps"] for t in rs])),
           ("Collisions per trial", lambda rs: np.mean([t["n_collisions"] for t in rs])),
           ("Commands blocked (%)", lambda rs: 100 * np.mean([t["override_rate"] for t in rs]))]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
for ax, (title, fn) in zip(axes, metrics):
    for i, ph in enumerate(PH):
        vals = [fn(nav[ph][g]) for g in COL]
        ax.bar(np.arange(2) + (i - 1) * w, vals, w * 0.94, color=list(COL.values()), alpha=ALPHA[ph],
               edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Healthy\nreference", "Stroke"])
    if "Collisions" in title:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "0 in every\ngroup and phase", ha="center", va="center", transform=ax.transAxes, fontsize=7)
    if "Goal" in title:
        ax.set_ylim(0, 110)
fig.tight_layout(w_pad=1.2)
fig.legend(h, ["Phase 1, no gain", "Phase 2, population reference", "Phase 3, subject reference"],
           loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
save(fig, "fig11_navigation.png")

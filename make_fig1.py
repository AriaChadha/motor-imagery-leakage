"""make_fig1.py - redraw Fig. 1 with panel (c) from outputs/leakage_scope.csv. Usage: python make_fig1.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
UNCTRL, CTRL = "#e15759", "#4e79a7"

df = pd.read_csv(os.path.join(ROOT, "outputs", "leakage_scope.csv"))
df = df.drop_duplicates(subset=["config", "trials_per_subject"], keep="last")   # latest run wins
paper = df[df.config == "paper"].sort_values("trials_per_subject")
ref = paper[paper.trials_per_subject == 38]
if ref.empty:
    raise SystemExit("no 'paper' row at 38 trials: run  python leakage_scope.py --config paper --trials 38")
ref = ref.iloc[0]
if ref.repeats != 20:
    print(f"!! the 38-trial row used {ref.repeats} repeats, not 20: rerun before making the final figure")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.9, 1.4]})

# ---- (a) within-subject
ax = axes[0]
m = [ref.within_out_mean, ref.within_in_mean]
s = [ref.within_out_sd, ref.within_in_sd]
bars = ax.bar(["CSP fitted\nbefore split", "CSP fitted\ninside fold"], m, yerr=s, capsize=4,
              color=[UNCTRL, CTRL], edgecolor="black", linewidth=0.6)
ax.axhline(0.5, color="black", ls=":", lw=1.3, zorder=0)
ax.text(0.015, 0.5, "chance 0.50", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8)
ax.annotate(f"+{(m[0] - 0.5) * 100:.1f} pp above chance", xy=(0, m[0]), xytext=(0, m[0] + 0.045), ha="center",
            fontsize=10, fontweight="bold", color=UNCTRL)
ax.set_title("(a) Within-subject regime\nbalanced labels, 38 trials", fontweight="bold", fontsize=10.5)
ax.set_ylabel("Accuracy on label-free data")
ax.set_ylim(0.44, max(m) + 0.10)

# ---- (b) cross-subject
ax = axes[1]
keys = ["cross_out_strat", "cross_out_group", "cross_in_strat", "cross_in_group"]
labels = ["CSP before split\npooled stratified", "CSP before split\ngrouped by subject",
          "CSP inside fold\npooled stratified", "CSP inside fold\ngrouped by subject"]
m2 = [ref[k + "_mean"] for k in keys]
s2 = [ref[k + "_sd"] for k in keys]
bars2 = ax.bar(labels, m2, yerr=s2, capsize=4, color=[UNCTRL] * 3 + [CTRL], edgecolor="black", linewidth=0.6)
ctrl, base = ref.cross_in_group_mean, ref.baseline_mean
ax.axhline(ctrl, color=CTRL, lw=1.4, zorder=0)
ax.text(0.015, ctrl, f"controlled protocol {ctrl:.3f}", transform=ax.get_yaxis_transform(),
        va="bottom", fontsize=8, color=CTRL)
ax.axhline(base, color="darkgreen", ls="--", lw=1.2, zorder=0)
ax.text(0.985, base, f"majority-class ceiling {base:.3f}", transform=ax.get_yaxis_transform(),
        va="bottom", ha="right", fontsize=8, color="darkgreen")
for b, v, e in zip(bars2[:3], m2[:3], s2[:3]):
    ax.text(b.get_x() + b.get_width() / 2, v + e + 0.006, f"+{(v - ctrl) * 100:.1f} pp", ha="center",
            fontsize=9.5, fontweight="bold", color=UNCTRL)
ax.set_title("(b) Cross-subject regime\ninflation relative to the controlled protocol",
             fontweight="bold", fontsize=10.5)
ax.set_ylim(0.44, max(max(v + e for v, e in zip(m2, s2)), base) + 0.04)
ax.tick_params(axis="x", labelsize=8.5)

# ---- (c) trial-count dependence
ax = axes[2]
x = paper.trials_per_subject.values
ax.errorbar(x, paper.within_out_mean, yerr=paper.within_out_sd, color=UNCTRL, marker="o",
            capsize=3, label="CSP fitted before split")
ax.errorbar(x, paper.within_in_mean, yerr=paper.within_in_sd, color=CTRL, marker="o",
            capsize=3, label="CSP fitted inside fold")
for xi, row in zip(x, paper.itertuples()):
    ax.text(xi, row.within_out_mean + 0.03, f"+{(row.within_out_mean - 0.5) * 100:.1f}", ha="center",
            fontsize=8.5, fontweight="bold", color=UNCTRL)
other = {"project": ("s", "real-data CSP, 300 trees"), "riemann": ("D", "tangent space + LR")}
for cfg, (mk, lab) in other.items():
    r = df[(df.config == cfg) & (df.trials_per_subject == 38)]
    if not r.empty:
        r = r.iloc[0]
        ax.scatter([38, 38], [r.within_out_mean, r.within_in_mean], marker=mk, s=45,
                   facecolors="none", edgecolors=[UNCTRL, CTRL], zorder=5)
        ax.scatter([], [], marker=mk, facecolors="none", edgecolors="black",
                   label=f"{lab} (38 trials): +{(r.within_out_mean - 0.5) * 100:.1f} pp above chance")
ax.axhline(0.5, color="black", ls=":", lw=1.3, zorder=0)
ax.set_xscale("log", base=2)
ax.set_xticks(x)
ax.set_xticklabels([str(int(v)) for v in x])
ax.set_xlabel("Trials per subject (22 channels)")
ax.set_title("(c) Within-subject inflation by trials per subject\n(labels: pp above chance, CSP before split)", fontweight="bold", fontsize=10.5)
ax.set_ylim(0.44, 1.02)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)

for a in axes:
    a.grid(axis="y", alpha=0.25)
    a.set_axisbelow(True)
fig.suptitle("Accuracy manufactured by evaluation protocol on data containing no class information",
             fontweight="bold", y=0.995, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", "fig1_leakage.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)

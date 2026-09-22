"""
evaluation_07.py — figures, statistics and CSV export

Two changes matter relative to the earlier version.

Navigation results are keyed by phase as well as by group:
    navigation = {phase: {group: [TrialResult, ...]}}
Previously a single group-keyed dictionary was passed in, so whichever phase
was written last overwrote the other and the navigation columns in results.csv
described only one condition. Collisions, goal completion, blocking rate and
latency are now reported per phase, which is what makes the Phase 1 to Phase 2
navigation comparison possible at all.

The group list is healthy and stroke. The Parkinson's group was dropped from
the design, and leaving it in produced empty bars and zero-filled rows that
looked like measurements.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats

try:
    from src.navigation_sim import TrialResult
except ModuleNotFoundError:
    from navigation_sim import TrialResult

GROUPS = ["healthy", "stroke"]
GROUP_COLORS = {"healthy": "#2196F3", "stroke": "#F44336"}
PHASE_HATCHES = {"Phase1": "", "Phase2": "//", "Phase3": "xx"}
CONTROL = "Phase1"
PHASE_LABELS = {"Phase1": "No amplification",
                "Phase2": "Population-referenced",
                "Phase3": "Subject-referenced"}


def _acc(phase_results, ph, g, key="rf_mean", default=np.nan):
    return phase_results.get(ph, {}).get(g, {}).get(key, default)


def _trials(phase_results, ph, g):
    return list(phase_results.get(ph, {}).get(g, {}).get("rf_trials", []))


# ── 1. Accuracy by group and phase ───────────────────────────────────────────
def plot_accuracy_by_phase(phase_results: dict, fig_dir: str) -> str:
    phases = list(phase_results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(GROUPS))
    for i, ph in enumerate(phases):
        means = [_acc(phase_results, ph, g, "rf_mean", 0.0) for g in GROUPS]
        stds = [_acc(phase_results, ph, g, "rf_std", 0.0) for g in GROUPS]
        ax.bar(x + (i - (len(phases) - 1) / 2) * width, means, width,
               yerr=stds, capsize=4,
               color=[GROUP_COLORS[g] for g in GROUPS],
               alpha=0.9 - 0.25 * i, hatch=PHASE_HATCHES.get(ph, ""),
               edgecolor="white")
    ax.axhline(0.5, color="grey", ls=":", lw=1.2, label="chance (2 classes)")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in GROUPS], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Within-subject accuracy", fontsize=11)
    ax.set_title("Decoding accuracy by group and phase",
                 fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(facecolor="grey", hatch=PHASE_HATCHES.get(p, ""),
                              alpha=0.8, label=p) for p in phases]
    handles += [mpatches.Patch(color=GROUP_COLORS[g], label=g.capitalize())
                for g in GROUPS]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(fig_dir, "accuracy_by_phase.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 2. Within-subject vs cross-subject ───────────────────────────────────────
def plot_within_vs_cross(phase_results: dict, fig_dir: str) -> str:
    """The gap between these two is the transfer penalty, a headline result."""
    ph = "Phase1" if "Phase1" in phase_results else list(phase_results)[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(GROUPS))
    w = 0.35
    within = [phase_results[ph].get(g, {}).get("within_subject", {}).get("mean", np.nan)
              for g in GROUPS]
    cross = [phase_results[ph].get(g, {}).get("cross_subject", {}).get("mean", np.nan)
             for g in GROUPS]
    ax.bar(x - w / 2, within, w, label="Within-subject (per-user calibration)",
           color=[GROUP_COLORS[g] for g in GROUPS], alpha=0.9)
    ax.bar(x + w / 2, cross, w, label="Cross-subject (new user)",
           color=[GROUP_COLORS[g] for g in GROUPS], alpha=0.45, hatch="//")
    ax.axhline(0.5, color="grey", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in GROUPS])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-user calibration vs transfer to a new user ({ph})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(fig_dir, "within_vs_cross.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 3. Amplification effect ──────────────────────────────────────────────────
def plot_improvement_delta(phase_results: dict, fig_dir: str) -> str:
    if "Phase1" not in phase_results or "Phase2" not in phase_results:
        return ""
    fig, ax = plt.subplots(figsize=(7, 5))
    deltas, labels, colors = [], [], []
    for g in GROUPS:
        p1 = _acc(phase_results, "Phase1", g, "rf_mean", np.nan)
        p2 = _acc(phase_results, "Phase2", g, "rf_mean", np.nan)
        deltas.append((p2 - p1) * 100)
        labels.append(g.capitalize())
        colors.append(GROUP_COLORS[g])
    bars = ax.bar(labels, deltas, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.9)
    for bar, d in zip(bars, deltas):
        if np.isfinite(d):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    d + (0.05 if d >= 0 else -0.15),
                    f"{d:+.2f} pp", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy change (percentage points)", fontsize=11)
    ax.set_title("Effect of adaptive amplification, Phase 1 to Phase 2",
                 fontsize=11, fontweight="bold")
    finite = [d for d in deltas if np.isfinite(d)]
    if finite:
        ax.set_ylim(min(0, min(finite) * 1.5), max(max(finite) * 1.45, 0.05))
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(fig_dir, "improvement_delta.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 4. Amplification factor distribution ─────────────────────────────────────
def plot_amplification_dist(factors: dict, fig_dir: str) -> str:
    amped_groups = [g for g in GROUPS
                    if np.any(np.asarray(factors.get(g, [1.0])) > 1.0)]
    if not amped_groups:
        return ""
    fig, axes = plt.subplots(1, len(amped_groups),
                             figsize=(5 * len(amped_groups), 4), squeeze=False)
    for ax, g in zip(axes[0], amped_groups):
        f = np.asarray(factors[g], dtype=float)
        fa = f[f > 1.0]
        ax.hist(fa, bins=25, color=GROUP_COLORS[g], edgecolor="white", alpha=0.85)
        ax.set_title(f"{g.capitalize()}: {len(fa)} of {len(f)} trials amplified",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Amplification factor")
        ax.set_ylabel("Trials")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Amplification factors, subject reference",
                 fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "amplification_dist.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 5. Per-subject accuracy, paired across phases ────────────────────────────
def plot_per_subject_paired(phase_results: dict, fig_dir: str) -> str:
    """Each line is one subject. Paired data is the basis for the paired test."""
    if "Phase1" not in phase_results or "Phase2" not in phase_results:
        return ""
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(5 * len(GROUPS), 5),
                             squeeze=False, sharey=True)
    for ax, g in zip(axes[0], GROUPS):
        p1 = phase_results["Phase1"].get(g, {}).get("within_subject", {}).get("per_subject", {})
        p2 = phase_results["Phase2"].get(g, {}).get("within_subject", {}).get("per_subject", {})
        shared = sorted(set(p1) & set(p2))
        for sid in shared:
            ax.plot([0, 1], [p1[sid], p2[sid]], "o-",
                    color=GROUP_COLORS[g], alpha=0.4, ms=4, lw=1)
        if shared:
            ax.plot([0, 1],
                    [np.mean([p1[s] for s in shared]),
                     np.mean([p2[s] for s in shared])],
                    "o-", color="black", lw=2.5, ms=8, label="mean")
            ax.legend(fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Phase 1\nno amplification", "Phase 2\namplified"])
        ax.set_title(f"{g.capitalize()} ({len(shared)} subjects)",
                     fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Within-subject accuracy")
    axes[0][0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.suptitle("Per-subject accuracy, paired across phases",
                 fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "per_subject_paired.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 6. Navigation: goal completion, collisions, blocking rate ────────────────
def plot_navigation(navigation: Dict[str, Dict[str, List[TrialResult]]],
                    fig_dir: str) -> str:
    phases = [p for p in navigation if navigation[p]]
    if not phases:
        return ""
    metrics = [
        ("Goal completion (%)", lambda rs: 100.0 * np.mean([r.reached_goal for r in rs])),
        ("Steps to goal", lambda rs: float(np.mean([r.n_steps for r in rs]))),
        ("Collisions per trial", lambda rs: float(np.mean([r.n_collisions for r in rs]))),
        ("Commands blocked (%)", lambda rs: 100.0 * np.mean([r.override_rate for r in rs])),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    x = np.arange(len(GROUPS))
    w = 0.35
    for ax, (title, fn) in zip(axes, metrics):
        for i, ph in enumerate(phases):
            vals = []
            for g in GROUPS:
                rs = navigation[ph].get(g, [])
                vals.append(fn(rs) if rs else np.nan)
            ax.bar(x + (i - (len(phases) - 1) / 2) * w, vals, w,
                   color=[GROUP_COLORS[g] for g in GROUPS],
                   alpha=0.9 - 0.3 * i, hatch=PHASE_HATCHES.get(ph, ""),
                   edgecolor="white", label=ph if ax is axes[0] else "")
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in GROUPS], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if "Collisions" in title:
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.suptitle("Navigation outcomes by group and phase",
                 fontsize=12, fontweight="bold")
    path = os.path.join(fig_dir, "navigation.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 7. Latency distribution ──────────────────────────────────────────────────
def plot_latency_dist(navigation: Dict[str, Dict[str, List[TrialResult]]],
                      fig_dir: str) -> str:
    data, lbls = [], []
    for ph in navigation:
        for g in GROUPS:
            rs = navigation[ph].get(g, [])
            lats = [1000 * l for r in rs for l in r.latencies]
            if lats:
                data.append(lats)
                lbls.append(f"{g[:4]}\n{ph}")
    if not data:
        return ""
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=lbls, patch_artist=True, showfliers=False)
    for patch, lbl in zip(bp["boxes"], lbls):
        g = "healthy" if lbl.startswith("heal") else "stroke"
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.7)
    ax.set_title("Command latency distribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(fig_dir, "latency_dist.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ── 8. Statistics ────────────────────────────────────────────────────────────
def run_statistics(phase_results: dict) -> dict:
    """Paired test on per-subject accuracies, each phase against the control.

    Paired rather than independent: the same subjects appear in every phase,
    and pairing removes between-subject variance, which is far larger than the
    amplification effect. Returns keys of the form "Phase2|stroke".

    The confidence interval matters more than the p-value. A p above 0.05 says
    only that no effect was detected; the interval says how large an effect the
    data can still accommodate, which is the claim a null is entitled to make.
    """
    out = {}
    if CONTROL not in phase_results:
        return out
    for ph in phase_results:
        if ph == CONTROL:
            continue
        for g in GROUPS:
            p1 = phase_results[CONTROL].get(g, {}).get("within_subject", {}).get("per_subject", {})
            p2 = phase_results[ph].get(g, {}).get("within_subject", {}).get("per_subject", {})
            shared = sorted(set(p1) & set(p2))
            if len(shared) < 3:
                continue
            v1 = np.array([p1[s] for s in shared])
            v2 = np.array([p2[s] for s in shared])
            diff = v2 - v1
            if np.allclose(diff, 0.0):
                t, p, w_p, d, lo, hi = 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
            else:
                t, p = stats.ttest_rel(v2, v1)
                w_p = stats.wilcoxon(v2, v1).pvalue
                sd = diff.std(ddof=1)
                d = float(diff.mean() / (sd + 1e-12))
                se = sd / np.sqrt(len(diff))
                crit = stats.t.ppf(0.975, len(diff) - 1)
                lo = float(100 * (diff.mean() - crit * se))
                hi = float(100 * (diff.mean() + crit * se))
            out[f"{ph}|{g}"] = {
                "phase": ph, "group": g, "n_subjects": len(shared),
                "mean_phase1": float(v1.mean()),
                "mean_phase2": float(v2.mean()),
                "mean_change_pp": float(100 * diff.mean()),
                "ci_low_pp": lo, "ci_high_pp": hi,
                "t": float(t), "p_paired_t": float(p),
                "p_wilcoxon": float(w_p), "cohens_dz": d,
            }
    return out

def run_data_analysis(phase_results: dict, factors: dict,
                      navigation: dict, stats_out: dict, out_dir: str) -> str:
    L = ["=" * 66, "DATA ANALYSIS SUMMARY", "=" * 66]

    L.append("\n--- Decoding accuracy ---")
    for ph in phase_results:
        L.append(f"\n{ph}:")
        for g in GROUPS:
            e = phase_results[ph].get(g, {})
            if not e:
                continue
            w = e.get("within_subject", {})
            c = e.get("cross_subject", {})
            L.append(f"  {g:<9} within {w.get('mean', float('nan')):.4f} "
                     f"+/- {w.get('std', float('nan')):.4f}   "
                     f"cross {c.get('mean', float('nan')):.4f} "
                     f"+/- {c.get('std', float('nan')):.4f}   "
                     f"n={e.get('n_trials', 0)} trials, "
                     f"{e.get('n_subjects', 0)} subjects")

    L.append("\n--- Amplification effect, paired by subject, vs " + CONTROL + " ---")
    if not stats_out:
        L.append("  not enough paired subjects for a test")
    for key, s in stats_out.items():
        verdict = "significant" if s["p_paired_t"] < 0.05 else "not significant"
        L.append(f"  {s['phase']}  {s['group']:<9} "
                 f"{s['mean_phase1']:.4f} -> {s['mean_phase2']:.4f}  "
                 f"({s['mean_change_pp']:+.2f} pp, 95% CI "
                 f"{s['ci_low_pp']:+.2f} to {s['ci_high_pp']:+.2f})  "
                 f"t={s['t']:+.3f}  p={s['p_paired_t']:.4f}  "
                 f"wilcoxon={s['p_wilcoxon']:.4f}  "
                 f"dz={s['cohens_dz']:+.3f}  n={s['n_subjects']}  [{verdict}]")

    L.append("\n--- Amplification factors ---")
    for g in GROUPS:
        f = np.asarray(factors.get(g, [1.0]), dtype=float)
        fa = f[f > 1.0]
        if fa.size:
            L.append(f"  {g:<9} {len(fa)}/{len(f)} trials amplified  "
                     f"mean={fa.mean():.4f}  sd={fa.std():.4f}  "
                     f"range {fa.min():.3f}-{fa.max():.3f}")
        else:
            L.append(f"  {g:<9} no trials amplified")

    L.append("\n--- Navigation ---")
    for ph in navigation:
        L.append(f"\n{ph}:")
        for g in GROUPS:
            rs = navigation[ph].get(g, [])
            if not rs:
                L.append(f"  {g:<9} no trials")
                continue
            L.append(f"  {g:<9} goal {100 * np.mean([r.reached_goal for r in rs]):5.1f}%  "
                     f"steps {np.mean([r.n_steps for r in rs]):6.1f}  "
                     f"collisions {np.mean([r.n_collisions for r in rs]):5.2f}  "
                     f"blocked {100 * np.mean([r.override_rate for r in rs]):5.1f}%  "
                     f"latency {np.mean([r.mean_lat_ms for r in rs]):5.0f} ms")

    L.append("\n" + "=" * 66)
    report = "\n".join(L)
    print(report)
    path = os.path.join(out_dir, "data_analysis.txt")
    with open(path, "w") as f:
        f.write(report)
    return path


# ── 9. CSV export ────────────────────────────────────────────────────────────
def export_csv(phase_results, navigation, factors, stats_out,
               out_path="outputs/results.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for ph, gdata in phase_results.items():
        for g, m in gdata.items():
            rs = navigation.get(ph, {}).get(g, [])
            f = np.asarray(factors.get((ph, g), [1.0]), dtype=float)
            rows.append({
                "phase": ph,
                "group": g,
                "n_trials": m.get("n_trials", 0),
                "n_subjects": m.get("n_subjects", 0),
                "within_mean": round(m.get("within_subject", {}).get("mean", np.nan), 4),
                "within_std": round(m.get("within_subject", {}).get("std", np.nan), 4),
                "cross_mean": round(m.get("cross_subject", {}).get("mean", np.nan), 4),
                "cross_std": round(m.get("cross_subject", {}).get("std", np.nan), 4),
                "mean_amp_factor": round(float(f.mean()), 4),
                "pct_amplified": round(100.0 * float((f > 1.0).mean()), 2),
                "sim_n": len(rs),
                "goal_pct": round(100 * np.mean([r.reached_goal for r in rs]), 1) if rs else np.nan,
                "mean_steps": round(float(np.mean([r.n_steps for r in rs])), 2) if rs else np.nan,
                "mean_collisions": round(float(np.mean([r.n_collisions for r in rs])), 3) if rs else np.nan,
                "blocked_pct": round(100 * np.mean([r.override_rate for r in rs]), 2) if rs else np.nan,
                "mean_latency_ms": round(float(np.mean([r.mean_lat_ms for r in rs])), 1) if rs else np.nan,
                "condition": PHASE_LABELS.get(ph, ph),
                "change_vs_control_pp": round(stats_out.get(f"{ph}|{g}", {}).get("mean_change_pp", np.nan), 3),
                "ci_low_pp": round(stats_out.get(f"{ph}|{g}", {}).get("ci_low_pp", np.nan), 3),
                "ci_high_pp": round(stats_out.get(f"{ph}|{g}", {}).get("ci_high_pp", np.nan), 3),
                "p_paired_t": round(stats_out.get(f"{ph}|{g}", {}).get("p_paired_t", np.nan), 4),
                "cohens_dz": round(stats_out.get(f"{ph}|{g}", {}).get("cohens_dz", np.nan), 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  results.csv -> {out_path}")
    return df


# ── Entry point ──────────────────────────────────────────────────────────────
def run_full_evaluation(phase_results: dict, navigation: dict, factors: dict,
                        out_dir: str = "outputs", data_raw=None) -> pd.DataFrame:
    """navigation must be {phase: {group: [TrialResult, ...]}}."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print("\n  Figures ...")

    flat = {g: v for (ph, g), v in factors.items() if ph == "Phase3"}

    plot_accuracy_by_phase(phase_results, fig_dir)
    plot_within_vs_cross(phase_results, fig_dir)
    plot_improvement_delta(phase_results, fig_dir)
    plot_amplification_dist(flat, fig_dir)
    plot_per_subject_paired(phase_results, fig_dir)
    plot_navigation(navigation, fig_dir)
    plot_latency_dist(navigation, fig_dir)

    stats_out = run_statistics(phase_results)
    run_data_analysis(phase_results, flat, navigation, stats_out, out_dir)

    import json
    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(stats_out, f, indent=2, default=float)

    return export_csv(phase_results, navigation, factors, stats_out,
                      os.path.join(out_dir, "results.csv"))
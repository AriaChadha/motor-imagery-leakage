import os
R = os.path.dirname(os.path.abspath(__file__))

def patch(rel, pairs):
    p = os.path.join(R, rel)
    if not os.path.exists(p): print(f"MISSING {rel}"); return
    s = open(p).read(); n = 0
    for old, new in pairs:
        if old in s: s = s.replace(old, new, 1); n += 1
        elif new in s: n += 1
        else: print(f"  ! no match in {rel}: {old.strip().splitlines()[0][:60]}")
    open(p, "w").write(s); print(f"  {rel}: {n}/{len(pairs)} edits applied")

def swap_func(rel, name, body):
    """Replace a whole top-level function, robust to comment drift."""
    p = os.path.join(R, rel)
    src = open(p).read()
    i = src.index(f"def {name}(")
    j = src.index("\ndef ", i + 1)
    open(p, "w").write(src[:i] + body.lstrip("\n") + src[j:])
    print(f"  {rel}: {name}() replaced")

print("Patching run_pipeline.py")
patch("run_pipeline.py", [
("""from amplification_03 import (apply_amplification, amplitude_compression,
                              clip_sensitivity, healthy_baseline,
                              snr_invariance_check, compute_factors)""",
 """from amplification_03 import (apply_amplification, amplitude_compression,
                              clip_sensitivity, healthy_baseline,
                              snr_invariance_check, compute_factors,
                              baseline_comparison, subject_baseline)"""),
("""def run_phase(data: Dict, phase: str, amplify: bool, seed: int,
              n_sim_trials: int, run_eegnet: bool, out_dir: str) -> Dict:""",
 """PHASE_LABELS = {"Phase1": "no amplification (control)",
                "Phase2": "amplification, population-referenced baseline",
                "Phase3": "amplification, subject-referenced baseline"}


def run_phase(data: Dict, phase: str, amplify: bool, seed: int,
              n_sim_trials: int, run_eegnet: bool, out_dir: str,
              mode: str = "population") -> Dict:"""),
("""    print(f" {phase} — amplification {'ENABLED' if amplify else 'DISABLED'}")
    print("=" * 70)

    data_phase, factors, baseline = apply_amplification(data, enabled=amplify)""",
 """    print(f" {phase} — {PHASE_LABELS.get(phase, phase)}")
    print("=" * 70)

    data_phase, factors, baseline = apply_amplification(
        data, enabled=amplify, mode=mode)"""),
("""    base = healthy_baseline(data["healthy"][0])
    Xs = data["stroke"][0]
    clip_sensitivity(Xs, base)
    amplitude_compression(Xs, base)
    snr_invariance_check(Xs, compute_factors(Xs, base))""",
 """    base = healthy_baseline(data["healthy"][0])
    Xs, _, Ss = data["stroke"]
    clip_sensitivity(Xs, base)
    amplitude_compression(Xs, base)
    snr_invariance_check(Xs, compute_factors(Xs, base))
    baseline_comparison(Xs, Ss, base)
    print("    subject-referenced compression:")
    amplitude_compression(Xs, subject_baseline(Xs, Ss))"""),
("""    phase2 = run_phase(data, "Phase2", amplify=True, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT)""",
 """    phase2 = run_phase(data, "Phase2", amplify=True, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT,
                       mode="population")
    phase3 = run_phase(data, "Phase3", amplify=True, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT,
                       mode="subject")"""),
("""    phase_results = {"Phase1": phase1["rf"], "Phase2": phase2["rf"]}
    navigation = {"Phase1": phase1["navigation"],
                  "Phase2": phase2["navigation"]}
    factors = phase2["factors"]          # Phase 1 factors are all 1.0""",
 """    phase_results = {"Phase1": phase1["rf"], "Phase2": phase2["rf"],
                     "Phase3": phase3["rf"]}
    navigation = {"Phase1": phase1["navigation"],
                  "Phase2": phase2["navigation"],
                  "Phase3": phase3["navigation"]}
    factors = phase3["factors"]     # subject-referenced, the adaptive condition"""),
("""        "eegnet": {"Phase1": phase1["eegnet"], "Phase2": phase2["eegnet"]},""",
 """        "eegnet": {"Phase1": phase1["eegnet"], "Phase2": phase2["eegnet"],
                   "Phase3": phase3["eegnet"]},"""),
("""        "accuracies": {"Phase1": phase1["accuracies"],
                       "Phase2": phase2["accuracies"]},
        "amplification_baseline": phase2["baseline"],""",
 """        "accuracies": {"Phase1": phase1["accuracies"],
                       "Phase2": phase2["accuracies"],
                       "Phase3": phase3["accuracies"]},
        "amplification_baseline": {"Phase2": phase2["baseline"],
                                   "Phase3": phase3["baseline"]},
        "phase2_factors": {g: v.tolist() for g, v in phase2["factors"].items()},"""),
])

print("Patching src/evaluation_07.py")
patch("src/evaluation_07.py", [
('PHASE_HATCHES = {"Phase1": "", "Phase2": "//"}',
 'PHASE_HATCHES = {"Phase1": "", "Phase2": "//", "Phase3": "xx"}\n'
 'CONTROL = "Phase1"\n'
 'PHASE_LABELS = {"Phase1": "No amplification",\n'
 '                "Phase2": "Population-referenced",\n'
 '                "Phase3": "Subject-referenced"}'),
("""    L.append("\\n--- Amplification effect, paired by subject ---")
    if not stats_out:
        L.append("  not enough paired subjects for a test")
    for g, s in stats_out.items():
        verdict = "significant" if s["p_paired_t"] < 0.05 else "not significant"
        L.append(f"  {g:<9} {s['mean_phase1']:.4f} -> {s['mean_phase2']:.4f}  "
                 f"({s['mean_change_pp']:+.2f} pp)  "
                 f"t={s['t']:+.3f}  p={s['p_paired_t']:.4f}  "
                 f"wilcoxon p={s['p_wilcoxon']:.4f}  "
                 f"dz={s['cohens_dz']:+.3f}  n={s['n_subjects']}  [{verdict}]")""",
 """    L.append("\\n--- Amplification effect, paired by subject, vs " + CONTROL + " ---")
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
                 f"dz={s['cohens_dz']:+.3f}  n={s['n_subjects']}  [{verdict}]")"""),
("""                "p_paired_t": round(stats_out.get(g, {}).get("p_paired_t", np.nan), 4),
                "cohens_dz": round(stats_out.get(g, {}).get("cohens_dz", np.nan), 4),""",
 """                "condition": PHASE_LABELS.get(ph, ph),
                "change_vs_control_pp": round(stats_out.get(f"{ph}|{g}", {}).get("mean_change_pp", np.nan), 3),
                "ci_low_pp": round(stats_out.get(f"{ph}|{g}", {}).get("ci_low_pp", np.nan), 3),
                "ci_high_pp": round(stats_out.get(f"{ph}|{g}", {}).get("ci_high_pp", np.nan), 3),
                "p_paired_t": round(stats_out.get(f"{ph}|{g}", {}).get("p_paired_t", np.nan), 4),
                "cohens_dz": round(stats_out.get(f"{ph}|{g}", {}).get("cohens_dz", np.nan), 4),"""),
])
swap_func("src/evaluation_07.py", "run_statistics", '''
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
''')

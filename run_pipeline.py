"""
run_pipeline.py — full experiment, healthy controls and post-stroke patients

  1. Load        data_loader_01    per-subject trial arrays with real labels
  2. Reject      preprocessing_02  artifact rejection, flatten to (X, y, subj)
  3. Phase 1     amplification off, classify, simulate navigation
  4. Phase 2     amplification on, same seeds, same code path
  5. Evaluate    evaluation_07     paired statistics, figures, results.csv
  6. Report      report_08         report.md, .html, .pdf

Both phases go through the same functions with the same seeds, so the only
difference between them is the amplification. Navigation is driven by the
measured within-subject accuracy of that phase; no navigation figure is
hardcoded anywhere.
"""

import os
import sys
import json
import time
import pickle
import argparse
from typing import Dict, Optional

import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader_01 import load_all
from preprocessing_02 import preprocess_all, rejection_report
from amplification_03 import (apply_amplification, amplitude_compression,
                              clip_sensitivity, healthy_baseline,
                              snr_invariance_check, compute_factors,
                              baseline_comparison, subject_baseline)
from classifier_rf_04 import (train_evaluate as rf_evaluate, feature_ablation,
                              permutation_test)
import navigation_sim as sim
from evaluation_07 import run_full_evaluation
from report_08 import build_report

OUT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(ROOT, "cache")
GROUPS = ["healthy", "stroke"]


# ----------------------------------------------------------------------

def prepare_data(use_cache: bool = True, max_subjects: Optional[int] = None,
                 ptp_threshold: float = 150.0) -> Dict:
    """Load and clean both groups. Returns {group: (X, y, subj)}."""
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{max_subjects or 'all'}_{ptp_threshold:g}"
    path = os.path.join(CACHE, f"clean_{tag}.npz")

    if use_cache and os.path.exists(path):
        print(f"[1-2] Cached data from {path}")
        d = np.load(path, allow_pickle=True)
        data = {g: (d[f"{g}_X"], d[f"{g}_y"], d[f"{g}_s"]) for g in GROUPS}
        for g, (X, y, s) in data.items():
            print(f"      {g}: {X.shape[0]} trials, {len(np.unique(s))} "
                  f"subjects, classes={np.bincount(y)}")
        return data

    print("[1] Loading datasets ...")
    groups = load_all(ROOT, max_subjects=max_subjects)

    print("\n[2] Artifact rejection ...")
    rejection_report(groups)
    data = preprocess_all(groups, ptp_threshold_uv=ptp_threshold)

    np.savez(path, **{f"{g}_{k}": v
                      for g, tup in data.items()
                      for k, v in zip(("X", "y", "s"), tup)})
    print(f"      cached -> {path}")
    return data


PHASE_LABELS = {"Phase1": "no amplification (control)",
                "Phase2": "amplification, population-referenced baseline",
                "Phase3": "amplification, subject-referenced baseline"}


def run_phase(data: Dict, phase: str, amplify: bool, seed: int,
              n_sim_trials: int, run_eegnet: bool, out_dir: str,
              mode: str = "population") -> Dict:
    """Amplify or not, classify, then simulate navigation."""
    print("\n" + "=" * 70)
    print(f" {phase} — {PHASE_LABELS.get(phase, phase)}")
    print("=" * 70)

    data_phase, factors, baseline = apply_amplification(
        data, enabled=amplify, mode=mode)

    print("\n  Random Forest:")
    rf = rf_evaluate(data_phase, phase=phase, out_dir=out_dir,
                     random_state=seed)

    eegnet = {}
    if run_eegnet:
        try:
            from classifier_eegnet_05 import train_evaluate as en_evaluate
            print("\n  EEGNet:")
            eegnet = en_evaluate(data_phase, phase=phase, out_dir=out_dir,
                                 seed=seed)
        except Exception as e:
            print(f"  EEGNet unavailable: {e}")

    # Navigation is driven by measured within-subject accuracy, the protocol a
    # per-user calibrated wheelchair BCI would actually use.
    accuracies = {}
    for g in GROUPS:
        acc = rf.get(g, {}).get("within_subject", {}).get("mean")
        if acc is not None and np.isfinite(acc):
            accuracies[g] = float(acc)

    navigation = {}
    if accuracies:
        print(f"\n  Navigation ({n_sim_trials} trials per group):")
        navigation = sim.run_all(
            accuracies, phase=phase, n_trials=n_sim_trials, seed=seed,
            out_json=os.path.join(out_dir, f"navigation_{phase}.json"))
    else:
        print("\n  No usable accuracies; navigation skipped.")

    return {"rf": rf, "eegnet": eegnet, "navigation": navigation,
            "factors": factors, "baseline": baseline,
            "accuracies": accuracies}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="EEG wheelchair pipeline (healthy + stroke)")
    ap.add_argument("--sim-trials", type=int, default=50,
                    help="navigation trials per group per phase")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="cap subjects per group")
    ap.add_argument("--ptp", type=float, default=400.0,
                    help="peak-to-peak rejection threshold in uV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eegnet", action="store_true",
                    help="also run EEGNet (slow)")
    ap.add_argument("--ablation", action="store_true",
                    help="run the feature ablation on stroke")
    ap.add_argument("--permutation", type=int, default=0, metavar="N",
                    help="run N label permutations per group to test whether "
                         "within-subject accuracy exceeds chance")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached preprocessed data")
    ap.add_argument("--author", type=str, default="",
                    help="name for the generated report")
    ap.add_argument("--smoke", action="store_true",
                    help="fast run: 5 subjects per group, 5 sim trials")
    args = ap.parse_args()

    if args.smoke:
        args.max_subjects = 5
        args.sim_trials = 5
        args.eegnet = False

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print(" EEG Wheelchair Control — healthy controls and post-stroke patients")
    print(f" seed {args.seed} | rejection {args.ptp:g} uV | "
          f"{args.sim_trials} sim trials"
          + (" | SMOKE TEST" if args.smoke else ""))
    print("=" * 70)

    data = prepare_data(use_cache=not args.no_cache,
                        max_subjects=args.max_subjects,
                        ptp_threshold=args.ptp)

    for g in GROUPS:
        if g not in data or data[g][0].shape[0] == 0:
            sys.exit(f"ERROR: no usable data for group '{g}'.")

    n_ch = {g: data[g][0].shape[1] for g in GROUPS}
    if len(set(n_ch.values())) > 1:
        sys.exit(f"ERROR: channel counts differ across groups: {n_ch}. "
                 f"Cross-population comparison would be invalid.")

    print("\n[3] Amplification diagnostics (stroke) ...")
    base = healthy_baseline(data["healthy"][0])
    Xs, _, Ss = data["stroke"]
    clip_sensitivity(Xs, base)
    amplitude_compression(Xs, base)
    snr_invariance_check(Xs, compute_factors(Xs, base))
    baseline_comparison(Xs, Ss, base)
    print("    subject-referenced compression:")
    amplitude_compression(Xs, subject_baseline(Xs, Ss))

    phase1 = run_phase(data, "Phase1", amplify=False, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT)
    phase2 = run_phase(data, "Phase2", amplify=True, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT,
                       mode="population")
    phase3 = run_phase(data, "Phase3", amplify=True, seed=args.seed,
                       n_sim_trials=args.sim_trials,
                       run_eegnet=args.eegnet, out_dir=OUT,
                       mode="subject")

    ablation = {}
    if args.ablation:
        print("\n[4] Feature ablation (stroke, Phase 1) ...")
        Xs, ys, ss = data["stroke"]
        ablation = feature_ablation(Xs, ys, ss, random_state=args.seed)

    perm = {}
    if args.permutation:
        print(f"\n[4b] Permutation test ({args.permutation} shuffles) ...")
        for g in GROUPS:
            print(f"  {g}:")
            Xg, yg, sg = data[g]
            perm[g] = permutation_test(Xg, yg, sg, random_state=args.seed,
                                       n_perm=args.permutation)

    phase_results = {"Phase1": phase1["rf"], "Phase2": phase2["rf"],
                     "Phase3": phase3["rf"]}
    navigation = {"Phase1": phase1["navigation"],
                  "Phase2": phase2["navigation"],
                  "Phase3": phase3["navigation"]}

    factors = {}
    for _ph, _res in [("Phase1", phase1), ("Phase2", phase2), ("Phase3", phase3)]:
        for _g, _v in _res["factors"].items():
            factors[(_ph, _g)] = _v

    print("\n[5] Evaluation ...")
    run_full_evaluation(phase_results, navigation, factors,
                        out_dir=OUT, data_raw=data)

    print("\n[6] Report ...")
    build_report(OUT, author=args.author)

    bundle = {
        "config": vars(args),
        "phase_results": phase_results,
        "eegnet": {"Phase1": phase1["eegnet"], "Phase2": phase2["eegnet"],
                   "Phase3": phase3["eegnet"]},
        "navigation": {ph: {g: [t.summary() for t in ts]
                            for g, ts in nav.items()}
                       for ph, nav in navigation.items()},
        "accuracies": {"Phase1": phase1["accuracies"],
                       "Phase2": phase2["accuracies"],
                       "Phase3": phase3["accuracies"]},
        "amplification_baseline": {"Phase2": phase2["baseline"],
                                   "Phase3": phase3["baseline"]},
        "phase2_factors": {g: v.tolist() for g, v in phase2["factors"].items()},
        "ablation": ablation,
        "permutation": perm,
    }
    with open(os.path.join(OUT, "pipeline_results.json"), "w") as f:
        json.dump(bundle, f, indent=2, default=float)
    with open(os.path.join(OUT, "pipeline_results.pkl"), "wb") as f:
        pickle.dump({**bundle, "factors": factors,
                     "navigation_raw": navigation}, f)

    print("\n" + "=" * 70)
    print(f" Complete in {time.time() - t0:.0f}s")
    print(f" Outputs in {OUT}")
    print("   report.md / report.html / report.pdf")
    print("   results.csv, statistics.json, data_analysis.txt")
    print("   figures/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
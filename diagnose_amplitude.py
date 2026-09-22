"""
diagnose_amplitude.py — why are the two groups on different amplitude scales?

Run from the project root:
    python diagnose_amplitude.py

Two candidate explanations, which call for different fixes:

  A. Global scale mismatch. The two datasets are in different units or were
     recorded with different amplifier gain. Every channel in one group sits
     above every channel in the other by roughly a constant ratio. If so, the
     amplification baseline is not physiologically meaningful and the method
     needs a within-dataset reference instead of a cross-dataset one.

  B. Frontal blink contamination. FP1, FP2, F7 and F8 pick up eye blinks,
     which reach several hundred microvolts. Rejection uses the worst channel
     over a 4 s window, so a single blink kills the trial. The stroke dataset
     supplies EOG channels that were dropped during loading, and its recordings
     may already be EOG-corrected, which would leave the healthy group looking
     far noisier than it is.

The two are distinguishable: under A the ratio is flat across channels, under B
it is concentrated in the frontal ones.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader_01 import load_all, CANONICAL_CHANS

FRONTAL = ["FP1", "FP2", "F7", "F8"]


def main(max_subjects: int = 8):
    groups = load_all(ROOT, max_subjects=max_subjects)

    stacked = {}
    for name, subs in groups.items():
        X = np.concatenate([s["X"] for s in subs])
        stacked[name] = X
        print(f"\n{name}: {X.shape[0]} trials, {X.shape[1]} channels, "
              f"{X.shape[2]} samples")
        print(f"  overall RMS      {np.sqrt((X.astype(np.float64) ** 2).mean()):9.3f} uV")
        print(f"  median |value|   {np.median(np.abs(X)):9.3f} uV")
        print(f"  99th pct |value| {np.percentile(np.abs(X), 99):9.3f} uV")

    h, s = stacked["healthy"], stacked["stroke"]

    # Per-channel peak-to-peak, median across trials.
    ph = np.median(h.max(axis=2) - h.min(axis=2), axis=0)
    ps = np.median(s.max(axis=2) - s.min(axis=2), axis=0)

    print("\nPer-channel median peak-to-peak (uV)")
    print(f"  {'channel':<9}{'healthy':>10}{'stroke':>10}{'ratio':>9}")
    ratios = []
    for i, ch in enumerate(CANONICAL_CHANS):
        r = ph[i] / max(ps[i], 1e-9)
        ratios.append(r)
        mark = "  <- frontal" if ch in FRONTAL else ""
        print(f"  {ch:<9}{ph[i]:>10.1f}{ps[i]:>10.1f}{r:>9.2f}{mark}")

    ratios = np.array(ratios)
    front_idx = [CANONICAL_CHANS.index(c) for c in FRONTAL
                 if c in CANONICAL_CHANS]
    rest_idx = [i for i in range(len(CANONICAL_CHANS)) if i not in front_idx]

    print("\nSummary")
    print(f"  ratio, frontal channels     {ratios[front_idx].mean():6.2f}")
    print(f"  ratio, all other channels   {ratios[rest_idx].mean():6.2f}")
    print(f"  ratio, spread across all    {ratios.min():6.2f} to {ratios.max():6.2f}")

    flat = ratios.std() / ratios.mean() < 0.35
    frontal_heavy = ratios[front_idx].mean() > 1.8 * ratios[rest_idx].mean()

    print("\nReading")
    if frontal_heavy:
        print("  Frontal channels dominate the gap, so blink contamination is")
        print("  the main driver. Options: drop FP1, FP2, F7, F8 from the")
        print("  montage, or reject per channel rather than on the worst")
        print("  channel across the whole montage.")
    if flat and not frontal_heavy:
        print("  The ratio is roughly constant across the montage, which points")
        print("  to a global scale or unit difference rather than to artifacts.")
        print("  The healthy-group RMS baseline is then not a physiologically")
        print("  meaningful reference for the stroke group.")
    if not flat and not frontal_heavy:
        print("  The gap is uneven but not frontal. Inspect the largest-ratio")
        print("  channels above individually before changing the montage.")

    # What a per-channel criterion would reject, for comparison.
    print("\nRejection rate under two criteria, 150 uV")
    for name, X in stacked.items():
        ptp = X.max(axis=2) - X.min(axis=2)
        worst = (ptp.max(axis=1) > 150).mean()
        no_frontal = (np.delete(ptp, front_idx, axis=1).max(axis=1) > 150).mean()
        print(f"  {name:<9} worst channel {100 * worst:5.1f}%    "
              f"excluding frontal {100 * no_frontal:5.1f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)

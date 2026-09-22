"""
preprocessing_02.py — re-referencing and artifact rejection (healthy + stroke)

Filtering, resampling, channel harmonization, epoching and label extraction all
happen in data_loader_01.py, where the dataset-specific details live. What
remains here is re-referencing and trial-level artifact rejection.

Re-referencing
--------------
The two datasets do not share a reference. The stroke recordings are referenced
to CPz, which suppresses amplitude at electrodes near the vertex: measured
median peak-to-peak at CZ was 7.6 times lower than in the healthy group, and
6.7 times lower at PZ, while peripheral sites such as T8 and TP8 differed by
only about 2 times. That spatial gradient is centred on the reference, not on
sensorimotor cortex, so it reflects the montage rather than stroke physiology.

Both groups are therefore re-referenced to their own common average before
anything else. Without this step the amplification factor, which is derived
from a healthy-versus-stroke RMS comparison, largely measures the difference in
reference scheme between two laboratories.

Amplitude is re-referenced but NOT normalized. The amplification experiment
manipulates signal scale directly, so rescaling trials to unit variance here
would erase the effect under study. Common average referencing does not do
that: it removes a shared component, and relative amplitude differences between
trials survive it.

Input:  {group: [{'id', 'X' (n_trials, 29, 640) uV, 'y'}, ...]}
Output: {group: (X, y, subj)} with rejected trials removed.
"""

from typing import Dict, Tuple, List

import numpy as np

# Peak-to-peak rejection threshold in microvolts, applied identically to both
# groups. A group-specific threshold would confound the comparison: a looser
# bound for stroke admits noisier trials, so any accuracy difference between
# groups would partly reflect the threshold rather than the physiology.
PTP_THRESHOLD_UV = 150.0

# Minimum surviving trials for a subject to be usable.
MIN_TRIALS = 10

# Liu et al. (Sci Data 11:131, 2024) flag motion artefacts in these stroke
# subjects. Off by default: rejection below is per-trial, which is less blunt
# than discarding whole subjects. Enable to test sensitivity.
FLAGGED_STROKE_SUBJECTS = {4, 5, 13, 14, 18, 24, 28, 33, 42, 43, 47, 48, 49}


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """Subtract the across-channel mean at every time point.

    X is (n_trials, n_channels, n_samples). The result is independent of the
    original reference electrode, up to the channels included in the average,
    which is what makes amplitudes comparable across the two datasets.

    One caveat worth stating in Methods: the common average is only as good as
    the montage it is computed over. With 29 electrodes and no full head
    coverage it is an approximation, and it is applied identically to both
    groups so it cannot bias the comparison in either direction.
    """
    return X - X.mean(axis=1, keepdims=True)


def reference_report(groups: Dict[str, List[dict]]) -> None:
    """Print per-group amplitude before and after re-referencing.

    Run this before preprocess_all to document the size of the reference
    effect. The before and after RMS values belong in the Methods section.
    """
    print("\n  Effect of common average referencing:")
    print(f"    {'group':<11}{'RMS before':>12}{'RMS after':>12}"
          f"{'p2p before':>12}{'p2p after':>12}")
    for group, subjects in groups.items():
        X = np.concatenate([s['X'] for s in subjects]).astype(np.float64)
        Xc = common_average_reference(X)
        r0 = np.sqrt((X ** 2).mean())
        r1 = np.sqrt((Xc ** 2).mean())
        p0 = np.median((X.max(axis=2) - X.min(axis=2)).max(axis=1))
        p1 = np.median((Xc.max(axis=2) - Xc.min(axis=2)).max(axis=1))
        print(f"    {group:<11}{r0:>11.2f}u{r1:>11.2f}u{p0:>11.1f}u{p1:>11.1f}u")


def preprocess_all(groups: Dict[str, List[dict]],
                   ptp_threshold_uv: float = PTP_THRESHOLD_UV,
                   min_trials: int = MIN_TRIALS,
                   exclude_flagged: bool = False,
                   rereference: bool = True
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Re-reference, reject high-amplitude trials, and flatten to arrays.

    Returns {group: (X, y, subj)} where X is (n_trials, 29, 640) float32 uV,
    y is int 0/1, and subj holds the subject id per trial so that later
    cross-validation can split by subject rather than by trial.

    rereference=False reproduces the original behaviour and exists so the
    sensitivity of the results to this choice can be reported rather than
    asserted.
    """
    out = {}
    print(f"  Common average reference: {'ON' if rereference else 'OFF'}")
    print(f"  Artifact rejection: peak-to-peak > {ptp_threshold_uv:g} uV")

    for group, subjects in groups.items():
        Xs, ys, ss = [], [], []
        n_seen = n_kept = 0
        dropped_subjects = []

        for subj in subjects:
            sid, X, y = subj['id'], subj['X'], subj['y']

            if exclude_flagged and group == 'stroke':
                num = int(sid.split('_')[-1])
                if num in FLAGGED_STROKE_SUBJECTS:
                    dropped_subjects.append(f"{sid} (flagged)")
                    continue

            # Re-reference before rejection. The threshold is an amplitude
            # criterion, so applying it to differently referenced data would
            # reject the two groups at incomparable rates, which is exactly
            # what happened before: 96% of healthy trials failed at 150 uV
            # against 37% of stroke trials.
            if rereference:
                X = common_average_reference(X.astype(np.float32))

            n_seen += len(y)
            keep = _clean_trials(X, ptp_threshold_uv)
            Xk, yk = X[keep], y[keep]

            if len(yk) < min_trials or len(np.unique(yk)) < 2:
                dropped_subjects.append(f"{sid} ({len(yk)} trials)")
                continue

            n_kept += len(yk)
            Xs.append(Xk.astype(np.float32))
            ys.append(yk)
            ss.append(np.full(len(yk), sid))

        if not Xs:
            raise RuntimeError(
                f"No usable {group} data after rejection. Every trial exceeded "
                f"{ptp_threshold_uv:g} uV peak-to-peak. Check the rejection "
                f"table above: if one group rejects far more than the other, "
                f"the two are on different amplitude scales and the threshold "
                f"is not the real problem.")

        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        s = np.concatenate(ss)
        out[group] = (X, y, s)

        pct = 100.0 * (n_seen - n_kept) / max(n_seen, 1)
        print(f"    {group}: {len(np.unique(s))} subjects, {n_kept} trials "
              f"({pct:.1f}% of trials rejected), classes={np.bincount(y)}")
        if dropped_subjects:
            print(f"      dropped: {', '.join(dropped_subjects)}")

    _warn_on_imbalance(out)
    _warn_on_amplitude_gap(out)
    return out


def _clean_trials(X: np.ndarray, threshold_uv: float) -> np.ndarray:
    """Boolean mask of trials whose worst channel stays under threshold.

    X is (n_trials, n_channels, n_samples) in microvolts. A trial is rejected
    if ANY channel exceeds the peak-to-peak bound, or if any value is
    non-finite.
    """
    ptp = X.max(axis=2) - X.min(axis=2)          # (n_trials, n_channels)
    finite = np.isfinite(X).all(axis=(1, 2))
    return (ptp.max(axis=1) <= threshold_uv) & finite


def _warn_on_imbalance(out: Dict, max_ratio: float = 2.0) -> None:
    """Flag a rejection-rate gap large enough to confound group comparison."""
    if len(out) < 2:
        return
    rates = {g: len(y) / max(len(np.unique(s)), 1)
             for g, (X, y, s) in out.items()}
    lo, hi = min(rates.values()), max(rates.values())
    if hi / max(lo, 1e-9) > max_ratio:
        print(f"      NOTE: trials per subject differ by more than "
              f"{max_ratio:g}x across groups {rates}. Report this in Methods; "
              f"it affects the per-subject accuracy comparison.")


def _warn_on_amplitude_gap(out: Dict, max_ratio: float = 1.5) -> None:
    """Flag a residual amplitude gap after re-referencing.

    A gap that survives common average referencing points to a difference in
    amplifier gain or units rather than to physiology, and it would propagate
    straight into the amplification factor.
    """
    if len(out) < 2:
        return
    rms = {g: float(np.sqrt((X.astype(np.float64) ** 2).mean()))
           for g, (X, y, s) in out.items()}
    lo, hi = min(rms.values()), max(rms.values())
    ratio = hi / max(lo, 1e-9)
    print(f"      Post-reference RMS by group: "
          f"{ {g: round(v, 2) for g, v in rms.items()} }  ratio {ratio:.2f}x")
    if ratio > max_ratio:
        print(f"      NOTE: a {ratio:.2f}x amplitude gap remains after common "
              f"average referencing. This is more likely amplifier gain or a "
              f"unit convention than physiology, and it feeds directly into "
              f"the amplification factor. Report it as a limitation, or "
              f"compute the amplification baseline within each dataset "
              f"instead of across them.")


def rejection_report(groups: Dict[str, List[dict]],
                     thresholds=(100., 150., 200., 300.),
                     rereference: bool = True) -> None:
    """Print rejection rate at several thresholds, per group.

    Useful for justifying the chosen threshold in the paper rather than
    asserting it. Uses the same referencing as preprocess_all, so the numbers
    printed here are the ones that will actually apply.
    """
    label = "after common average reference" if rereference else "raw"
    print(f"\n  Rejection rate by threshold (uV peak-to-peak, {label}):")
    print("    group      " + "".join(f"{t:>10.0f}" for t in thresholds))
    for group, subjects in groups.items():
        X = np.concatenate([s['X'] for s in subjects])
        if rereference:
            X = common_average_reference(X)
        ptp = (X.max(axis=2) - X.min(axis=2)).max(axis=1)
        row = "".join(f"{100.0 * (ptp > t).mean():>9.1f}%" for t in thresholds)
        print(f"    {group:<11}{row}")


if __name__ == "__main__":
    from data_loader_01 import load_all

    groups = load_all(".", max_subjects=8)
    reference_report(groups)
    rejection_report(groups, rereference=False)
    rejection_report(groups, rereference=True)
    data = preprocess_all(groups)
    for g, (X, y, s) in data.items():
        print(f"{g}: X={X.shape} y={np.bincount(y)} "
              f"subjects={len(np.unique(s))}")
"""
amplification_03.py - adaptive signal amplification for impaired EEG

    deficit = (baseline_rms - trial_rms) / baseline_rms
    factor  = clip(1.0 + deficit * GAIN, AMP_MIN, AMP_MAX)

Only trials below the baseline are amplified. Labels are never touched.

Two reference modes
-------------------
population  The baseline is the healthy group's mean trial RMS. This is the
            original formulation. It compares amplitudes across two datasets
            recorded in different laboratories on different amplifiers, so any
            instrumental gain difference reads as an amplitude deficit. In
            practice nearly every impaired trial falls below the baseline and
            the rule degenerates into a near-constant gain, which is no longer
            adaptive in any meaningful sense.

subject     The baseline is each subject's own median trial RMS. A trial is
            amplified when it is weak relative to that user's own typical
            amplitude, which is what an adaptive system would actually key on.
            About half of each subject's trials fall below their own median by
            construction, so the rule stays adaptive, and no cross-dataset
            amplitude comparison enters the calculation at all.

Running both is what separates "amplification does not help" from
"this particular baseline was mis-specified".
"""

from typing import Dict, Tuple, Optional

import numpy as np

GAIN    = 0.5
AMP_MIN = 1.10
AMP_MAX = 1.50

IMPAIRED = ("stroke",)
MODES = ("population", "subject")

Group = Tuple[np.ndarray, np.ndarray, np.ndarray]


def trial_rms(X: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per trial, in microvolts."""
    return np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=(1, 2)))


def healthy_baseline(X_healthy: np.ndarray) -> float:
    """Mean trial RMS of the healthy group."""
    if X_healthy.size == 0:
        raise ValueError("Healthy group is empty; cannot compute baseline.")
    r = trial_rms(X_healthy)
    b = float(np.mean(r))
    print(f"  Population baseline (healthy RMS): {b:.3f} uV "
          f"(median {np.median(r):.3f}, IQR "
          f"{np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f})")
    return b


def subject_baseline(X: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """Per-trial baseline equal to that trial's own subject median RMS."""
    r = trial_rms(X)
    out = np.empty_like(r)
    for sid in np.unique(subj):
        m = subj == sid
        out[m] = np.median(r[m])
    return out


def compute_factors(X: np.ndarray, baseline,
                    gain: float = GAIN,
                    amp_min: float = AMP_MIN,
                    amp_max: float = AMP_MAX) -> np.ndarray:
    """Per-trial amplification factor. baseline is a scalar or a per-trial array."""
    rms = trial_rms(X)
    b = np.broadcast_to(np.asarray(baseline, dtype=np.float64), rms.shape)
    factors = np.ones(len(X))
    below = rms < b
    factors[below] = np.clip(
        1.0 + (b[below] - rms[below]) / b[below] * gain, amp_min, amp_max)
    return factors


def amplify(X: np.ndarray, baseline, group: str,
            gain: float = GAIN, amp_min: float = AMP_MIN,
            amp_max: float = AMP_MAX) -> Tuple[np.ndarray, np.ndarray]:
    """Scale each below-baseline trial by its factor. Labels untouched."""
    f = compute_factors(X, baseline, gain, amp_min, amp_max)
    X_out = (X * f[:, None, None]).astype(X.dtype)
    amped = f > 1.0
    n = int(amped.sum())
    if n:
        fa = f[amped]
        print(f"  [{group}] amplified {n}/{len(X)} ({100.0 * n / len(X):.1f}%), "
              f"factor mean={fa.mean():.3f} min={fa.min():.3f} max={fa.max():.3f}")
        if np.isclose(fa.max(), amp_max):
            print(f"      note: upper clip {amp_max:g} is binding")
        if 100.0 * n / len(X) > 90.0:
            print(f"      note: over 90% of trials amplified, so the rule is "
                  f"close to a constant gain rather than an adaptive one")
    else:
        print(f"  [{group}] no trials below baseline; nothing amplified")
    return X_out, f


def apply_amplification(data: Dict[str, Group], enabled: bool = True,
                        mode: str = "population", gain: float = GAIN,
                        amp_min: float = AMP_MIN, amp_max: float = AMP_MAX):
    """Amplify the impaired groups. Returns (data, factors, baseline_info).

    enabled=False returns the data unchanged with unit factors, which is the
    Phase 1 control. Every phase runs through this same function so the only
    difference between them is the amplification itself.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if not enabled:
        print("\nAmplification DISABLED (control condition)")
        return dict(data), {g: np.ones(len(v[0])) for g, v in data.items()}, None

    print(f"\nAdaptive amplification, {mode}-referenced "
          f"(gain={gain:g}, clip=[{amp_min:g}, {amp_max:g}]) ...")

    base_info = {"mode": mode}
    if mode == "population":
        if "healthy" not in data:
            raise KeyError("Healthy group required for the population baseline.")
        pop = healthy_baseline(data["healthy"][0])
        base_info["population_rms"] = pop

    out, factors = {}, {}
    for g, (X, y, s) in data.items():
        if g not in IMPAIRED:
            out[g] = (X, y, s)
            factors[g] = np.ones(len(X))
            continue
        if mode == "population":
            b = pop
        else:
            b = subject_baseline(X, s)
            base_info[f"{g}_subject_rms_median"] = float(np.median(b))
            print(f"  Subject baselines for {g}: median "
                  f"{np.median(b):.3f} uV across {len(np.unique(s))} subjects")
        Xa, f = amplify(X, b, g, gain, amp_min, amp_max)
        out[g] = (Xa, y, s)
        factors[g] = f
    return out, factors, base_info


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def amplitude_compression(X: np.ndarray, baseline, **kw) -> None:
    """Does amplification narrow the spread of trial amplitudes?

    The factor is derived from trial RMS, so amplification pulls low-amplitude
    trials toward the baseline. A drop in the coefficient of variation means
    real amplitude variation has been reduced, which is a concrete mechanism
    for accuracy loss that does not rely on any argument about noise.
    """
    r0 = trial_rms(X)
    r1 = trial_rms(X * compute_factors(X, baseline, **kw)[:, None, None])
    cv0, cv1 = r0.std() / r0.mean(), r1.std() / r1.mean()
    print(f"    RMS spread: CV {cv0:.4f} -> {cv1:.4f} "
          f"({100.0 * (cv1 - cv0) / cv0:+.1f}%)")


def snr_invariance_check(X: np.ndarray, factors: np.ndarray) -> None:
    """Uniform scaling leaves every within-trial ratio unchanged."""
    i = int(np.argmax(factors))
    if factors[i] <= 1.0:
        print("    (no amplified trial available for the ratio check)")
        return
    a = X[i]
    ra = np.sqrt((a ** 2).mean(axis=1))
    rb = np.sqrt(((a * factors[i]) ** 2).mean(axis=1))
    rel = np.abs(rb / rb[0] - ra / ra[0]).max()
    print(f"    inter-channel amplitude ratios unchanged "
          f"(max deviation {rel:.2e}, factor {factors[i]:.3f})")


def clip_sensitivity(X: np.ndarray, baseline) -> None:
    """Compare AMP_MIN settings, to justify the choice in Methods."""
    print("    AMP_MIN sensitivity:")
    for amp_min in (1.0, 1.10):
        f = compute_factors(X, baseline, amp_min=amp_min)
        fa = f[f > 1.0]
        print(f"      amp_min={amp_min:4.2f}: "
              f"{100.0 * len(fa) / len(f):5.1f}% amplified, "
              f"mean factor {fa.mean() if len(fa) else 1.0:.4f}")


def baseline_comparison(X: np.ndarray, subj: np.ndarray,
                        population: float) -> None:
    """Show how differently the two reference modes behave on the same data."""
    print("    Reference mode comparison:")
    for name, b in (("population", population),
                    ("subject", subject_baseline(X, subj))):
        f = compute_factors(X, b)
        fa = f[f > 1.0]
        print(f"      {name:<11} {100.0 * len(fa) / len(f):5.1f}% amplified, "
              f"mean factor {fa.mean() if len(fa) else 1.0:.4f}")
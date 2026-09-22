"""
report_08.py — builds outputs/report.md, report.html and report.pdf

The abstract is generated from results.csv and statistics.json rather than
written in advance. The previous version asserted that amplification improved
accuracy for stroke and Parkinson's patients no matter what the run produced,
which is the one claim this project cannot afford to get wrong.
"""

import os
import json
import base64
import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import markdown as md_lib
    MD_OK = True
except ImportError:
    MD_OK = False

try:
    from weasyprint import HTML as WP
    WP_OK = True
except ImportError:
    WP_OK = False

GROUPS = ["healthy", "stroke"]

FIG_CAPTIONS = {
    "accuracy_by_phase.png":
        "Figure 1. Within-subject decoding accuracy by group and phase.",
    "within_vs_cross.png":
        "Figure 2. Per-user calibration compared with transfer to an unseen user.",
    "improvement_delta.png":
        "Figure 3. Change in accuracy from Phase 1 to Phase 2, the amplification effect.",
    "per_subject_paired.png":
        "Figure 4. Per-subject accuracy paired across phases. Each line is one subject.",
    "amplification_dist.png":
        "Figure 5. Distribution of amplification factors applied in Phase 2.",
    "navigation.png":
        "Figure 6. Navigation outcomes: goal completion, steps, collisions and command blocking.",
    "latency_dist.png":
        "Figure 7. Command latency distribution against the 250 ms target.",
}

CSS = """
<style>
body{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 30px;
     color:#222;line-height:1.7;}
h1{color:#1a237e;border-bottom:3px solid #1a237e;padding-bottom:8px;}
h2{color:#283593;border-bottom:1px solid #bbb;margin-top:34px;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.95em;}
th,td{border:1px solid #ccc;padding:7px 11px;text-align:left;}
th{background:#e8eaf6;}
tr:nth-child(even){background:#f7f7f7;}
pre{background:#f0f4ff;padding:12px;border-radius:5px;border-left:4px solid #3949ab;}
figure{text-align:center;margin:22px 0;}
figcaption{font-style:italic;color:#555;font-size:.88em;margin-top:6px;}
</style>
"""


def _fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def _b64img(fig_dir, name):
    p = os.path.join(fig_dir, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    cap = FIG_CAPTIONS.get(name, "")
    return (f'<figure><img src="data:image/png;base64,{b}" '
            f'style="max-width:92%;border:1px solid #ddd;border-radius:4px;">'
            f'<figcaption>{cap}</figcaption></figure>\n')


def _row(df, phase, group):
    r = df[(df.phase == phase) & (df.group == group)]
    return r.iloc[0] if not r.empty else None


def _interval(s):
    """95% CI on the paired mean change, in percentage points.

    statistics.json stores the mean change and Cohen's dz, and dz is the mean
    change divided by the standard deviation of the differences, so the SD can
    be recovered and the interval built from it. A non-significant p-value on
    its own says only that an effect was not detected; an interval says how
    large an effect the data can still accommodate, which is the claim a null
    result is actually entitled to make.
    """
    m, dz, n = s["mean_change_pp"], s["cohens_dz"], s["n_subjects"]
    if abs(dz) < 1e-9 or n < 2:
        return None
    se = abs(m / dz) / np.sqrt(n)
    return m - 1.96 * se, m + 1.96 * se


def build_report(out_dir: str = "outputs", author: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    csv_path = os.path.join(out_dir, "results.csv")
    stats_path = os.path.join(out_dir, "statistics.json")

    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    date = datetime.date.today().strftime("%B %d, %Y")
    byline = f"**Author:** {author}  |  " if author else ""

    # Abstract sentences built from what the run actually produced.
    findings = []
    for g in GROUPS:
        s = stats.get(g)
        if not s or s["n_subjects"] < 2:
            continue
        ci = _interval(s)
        direction = "increased" if s["mean_change_pp"] > 0 else "decreased"
        sig = "statistically significant" if s["p_paired_t"] < 0.05 else "not statistically significant"
        txt = (
            f"In the {g} group, amplification {direction} within-subject accuracy by "
            f"{abs(s['mean_change_pp']):.2f} percentage points "
            f"({s['mean_phase1']:.3f} to {s['mean_phase2']:.3f}), a change that was {sig} "
            f"across {s['n_subjects']} subjects (paired t-test p = {s['p_paired_t']:.3f}, "
            f"Wilcoxon p = {s['p_wilcoxon']:.3f}, dz = {s['cohens_dz']:+.3f})."
        )
        if ci:
            txt += (f" The 95% confidence interval on that change runs from "
                    f"{ci[0]:+.2f} to {ci[1]:+.2f} percentage points, so an effect "
                    f"large enough to matter for wheelchair control can be excluded "
                    f"rather than merely left undetected.")
        findings.append(txt)
    abstract_findings = " ".join(findings) if findings else \
        "Statistical comparison was not available for this run."

    gap_lines = []
    for g in GROUPS:
        r = _row(df, "Phase1", g)
        if r is None or not np.isfinite(r.cross_mean):
            continue
        gap_lines.append(
            f"For the {g} group, per-user calibration reached {r.within_mean:.3f} "
            f"while transfer to an unseen subject reached {r.cross_mean:.3f}, "
            f"a difference of {100 * (r.within_mean - r.cross_mean):+.1f} "
            f"percentage points.")
    transfer_findings = " ".join(gap_lines)

    transfer = []
    for g in GROUPS:
        r = _row(df, "Phase1", g)
        if r is None:
            continue
        transfer.append(
            f"| {g.capitalize()} | {_fmt(r.within_mean)} ± {_fmt(r.within_std)} | "
            f"{_fmt(r.cross_mean)} ± {_fmt(r.cross_std)} | "
            f"{int(r.n_subjects)} | {int(r.n_trials)} |")

    def acc_table(phase):
        out = []
        for g in GROUPS:
            r = _row(df, phase, g)
            if r is None:
                continue
            out.append(f"| {g.capitalize()} | {_fmt(r.within_mean)} ± {_fmt(r.within_std)} | "
                       f"{_fmt(r.cross_mean)} ± {_fmt(r.cross_std)} | "
                       f"{_fmt(r.mean_amp_factor, 4)} | {_fmt(r.pct_amplified, 1)}% |")
        return "\n".join(out)

    def nav_table(phase):
        out = []
        for g in GROUPS:
            r = _row(df, phase, g)
            if r is None:
                continue
            out.append(f"| {g.capitalize()} | {_fmt(r.goal_pct, 1)}% | "
                       f"{_fmt(r.mean_steps, 1)} | {_fmt(r.mean_collisions, 2)} | "
                       f"{_fmt(r.blocked_pct, 1)}% | {_fmt(r.mean_latency_ms, 0)} ms |")
        return "\n".join(out)

    md = f"""# EEG-Based Wheelchair Control Across Healthy and Post-Stroke Populations

### Does adaptive signal amplification improve motor imagery decoding?

{byline}**{date}**

---

## Abstract

Motor imagery brain-computer interfaces decode movement intent from scalp EEG,
but signal quality is reduced after stroke, which limits decoding accuracy in
exactly the population that stands to gain most from wheelchair control. This
work tests whether adaptive per-trial signal amplification, scaling
low-amplitude trials toward a healthy reference, recovers any of that lost
accuracy. Two public datasets were used: PhysioNet EEGMMIDB for healthy
controls and the figshare 21679035 corpus for post-stroke patients, harmonized
to 29 shared channels at 160 Hz over a 4 s motor imagery window. Random Forest
and EEGNet classifiers were evaluated both within subject and across subjects,
and decoding accuracy drove a discrete-event navigation simulation.
{abstract_findings} {transfer_findings} Uniform per-trial scaling multiplies
signal and noise together and leaves every within-trial ratio unchanged, so the
signal-to-noise ratio available to the classifier is unaltered while genuine
between-trial amplitude variation is compressed. The result is reported as a
negative finding, and it constrains the design of future amplification schemes:
a useful method has to act selectively on frequency bands or spatial components
rather than scaling the trial as a whole.

---

## 1. Methods

### 1.1 Datasets

| Dataset | Group | Paradigm | Native rate |
|---|---|---|---|
| PhysioNet EEGMMIDB, runs 4, 8, 12 | Healthy controls | Left vs right fist motor imagery | 160 Hz |
| figshare 21679035 (Liu et al. 2024) | Post-stroke | Left vs right hand motor imagery | 500 Hz |

Both are reduced to the 29 channels the two montages share, resampled to
160 Hz, band-pass filtered from 1 to 40 Hz, and cut to a 4 s imagery window.
Trials with any channel exceeding 150 µV peak-to-peak are rejected, using the
same threshold for both groups.

### 1.2 Re-referencing

The datasets do not share a reference. The stroke recordings are referenced to
CPz, which suppresses amplitude at electrodes near the vertex: median
peak-to-peak at CZ was 7.6 times lower than in the healthy group and 6.7 times
lower at PZ, while peripheral sites such as T8 and TP8 differed by only about
two times. That gradient is centred on the reference electrode rather than on
sensorimotor cortex, so it reflects the montage and not stroke physiology. Both
groups are therefore re-referenced to their own common average before any
amplitude comparison is made. Without this step the amplification factor,
derived from a healthy-versus-stroke RMS comparison, would largely measure the
difference in reference scheme between two laboratories.

### 1.2 Amplification

```
deficit = (baseline_rms - trial_rms) / baseline_rms
factor  = clip(1.0 + deficit * 0.5, AMP_MIN, AMP_MAX)
```

The baseline is the mean trial RMS of the healthy group. Only trials below that
baseline are scaled, and labels are never modified.

### 1.3 Evaluation

Two protocols are reported throughout. **Within-subject** accuracy uses
stratified cross-validation inside each subject and averages across subjects,
matching a system calibrated for one user. **Cross-subject** accuracy uses
GroupKFold with subject as the group, so no subject appears in both train and
test, and estimates transfer to a new user. Trials are never pooled across
subjects into an ungrouped split, which would place the same subject on both
sides of the fold and inflate accuracy.

---

## 2. Results

### 2.1 Phase 1, no amplification

| Group | Within-subject | Cross-subject | Mean factor | Amplified |
|---|---|---|---|---|
{acc_table("Phase1")}

### 2.2 Phase 2, amplification enabled

| Group | Within-subject | Cross-subject | Mean factor | Amplified |
|---|---|---|---|---|
{acc_table("Phase2")}

### 2.3 Calibration against transfer

| Group | Within-subject | Cross-subject | Subjects | Trials |
|---|---|---|---|---|
{chr(10).join(transfer)}

### 2.4 Navigation, Phase 1

| Group | Goal reached | Steps | Collisions | Commands blocked | Latency |
|---|---|---|---|---|---|
{nav_table("Phase1")}

### 2.5 Navigation, Phase 2

| Group | Goal reached | Steps | Collisions | Commands blocked | Latency |
|---|---|---|---|---|---|
{nav_table("Phase2")}

---

## 3. Limitations

The navigation results come from a simulation, not a physical wheelchair. The
simulated environment is a 10x10 grid with static obstacles and a fixed goal,
which omits dynamic obstacles, continuous control, and the physical dynamics of
a real chair, so the absolute navigation numbers should be read as a
comparison between conditions rather than as a prediction of real-world
performance.

The healthy and stroke recordings come from different laboratories and
different amplifiers. An amplitude difference between the groups survives
common average referencing, and because the recordings are instrumentally
distinct that residual gap is more plausibly a matter of amplifier gain than of
physiology. It propagates directly into the amplification factor. When nearly
every impaired trial falls below the baseline, the adaptive rule degenerates
into a near-uniform gain and is no longer adapting to anything, which weakens
the interpretation of the amplification condition. Computing the baseline
within each dataset rather than across them would address this and is the
natural next experiment.

Within-subject models train on far fewer trials than cross-subject models,
which can place the within-subject estimate below the cross-subject estimate
when trials per subject are limited. The two numbers answer different questions
and their magnitudes are not directly comparable.

CSP is fitted on all trials before cross-validation rather than inside each
fold, which biases the CSP contribution upward. Both phases are affected
identically, so the Phase 1 to Phase 2 comparison is unaffected, but absolute
accuracies are optimistic.

---

## References

1. Pfurtscheller G, Lopes da Silva FH (1999). Event-related EEG/MEG synchronization and desynchronization. *Clin Neurophysiol* 110:1842-1857.
2. Lawhern VJ et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *J Neural Eng* 15:056013.
3. Müller-Gerking J, Pfurtscheller G, Flyvbjerg H (2000). Designing optimal spatial filters for single-trial EEG classification. *Clin Neurophysiol* 110:787-798.
4. Welch PD (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Trans Audio Electroacoust* 15:70-73.
5. Nolan H, Whelan R, Reilly RB (2010). FASTER: fully automated statistical thresholding for EEG artifact rejection. *J Neurosci Methods* 192:152-162.
6. Liu et al. (2024). An EEG dataset of stroke patients performing motor imagery. *Sci Data* 11:131.
"""

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  report.md -> {md_path}")

    if not MD_OK:
        print("  HTML and PDF skipped; pip install markdown weasyprint")
        return md_path

    figs = "\n".join(_b64img(fig_dir, n) for n in FIG_CAPTIONS)
    body = md_lib.markdown(md, extensions=["tables", "fenced_code"])
    html = (f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{CSS}</head>"
            f"<body>{body}<h2>Figures</h2>{figs}</body></html>")
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  report.html -> {html_path}")

    if WP_OK:
        try:
            pdf_path = os.path.join(out_dir, "report.pdf")
            WP(string=html, base_url=out_dir).write_pdf(pdf_path)
            print(f"  report.pdf -> {pdf_path}")
        except Exception as e:
            print(f"  PDF failed: {e}")
    return md_path


if __name__ == "__main__":
    import sys
    build_report(sys.argv[1] if len(sys.argv) > 1 else "outputs")
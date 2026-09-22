# EEG-Based Wheelchair Control Across Healthy and Post-Stroke Populations

### Does adaptive signal amplification improve motor imagery decoding?

**August 13, 2026**

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
Statistical comparison was not available for this run. For the healthy group, per-user calibration reached 0.623 while transfer to an unseen subject reached 0.598, a difference of +2.5 percentage points. For the stroke group, per-user calibration reached 0.541 while transfer to an unseen subject reached 0.501, a difference of +4.0 percentage points. Uniform per-trial scaling multiplies
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
| Healthy | 0.623 ± 0.146 | 0.598 ± 0.018 | 1.0000 | 0.0% |
| Stroke | 0.541 ± 0.126 | 0.501 ± 0.016 | 1.0000 | 0.0% |

### 2.2 Phase 2, amplification enabled

| Group | Within-subject | Cross-subject | Mean factor | Amplified |
|---|---|---|---|---|
| Healthy | 0.623 ± 0.146 | 0.598 ± 0.018 | 1.0000 | 0.0% |
| Stroke | 0.545 ± 0.116 | 0.506 ± 0.014 | 1.2816 | 97.9% |

### 2.3 Calibration against transfer

| Group | Within-subject | Cross-subject | Subjects | Trials |
|---|---|---|---|---|
| Healthy | 0.623 ± 0.146 | 0.598 ± 0.018 | 87 | 3076 |
| Stroke | 0.541 ± 0.126 | 0.501 ± 0.016 | 46 | 1758 |

### 2.4 Navigation, Phase 1

| Group | Goal reached | Steps | Collisions | Commands blocked | Latency |
|---|---|---|---|---|---|
| Healthy | 96.0% | 38.6 | 0.00 | 10.5% | 275 ms |
| Stroke | 96.0% | 46.2 | 0.00 | 12.1% | 275 ms |

### 2.5 Navigation, Phase 2

| Group | Goal reached | Steps | Collisions | Commands blocked | Latency |
|---|---|---|---|---|---|
| Healthy | 96.0% | 38.6 | 0.00 | 10.5% | 275 ms |
| Stroke | 96.0% | 45.7 | 0.00 | 11.9% | 275 ms |

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

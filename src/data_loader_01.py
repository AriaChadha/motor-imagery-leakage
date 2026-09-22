"""
data_loader_01.py — Load EEG datasets (two-group design: healthy + stroke)

Returns trial-segmented arrays, not continuous recordings, because the stroke
labels exist only in the sourcedata .mat files as per-trial annotations.

Both groups are returned on a common footing:
    29 EEG channels, 160 Hz, 4 s motor-imagery window, microvolts,
    class 0 = left hand, class 1 = right hand.
"""

import os
import re
import gc
import glob
from typing import Dict, List, Optional

import numpy as np
import mne
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample_poly

# ----------------------------------------------------------------------
# Common target format
# ----------------------------------------------------------------------

TARGET_FS = 160.0          # PhysioNet native rate; stroke is downsampled to it
WIN_SEC   = 4.0            # motor-imagery window in both datasets
N_SAMP    = int(TARGET_FS * WIN_SEC)      # 640
BAND      = (1.0, 40.0)

# 29 EEG channels shared by both datasets.
# CPz is excluded: it is the reference in the stroke recordings and carries
# no independent signal. EOG and marker channels are excluded as well.
CANONICAL_CHANS = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8',
    'FCZ', 'FC3', 'FC4', 'FT7', 'FT8',
    'CZ', 'C3', 'C4', 'T7', 'T8',
    'CP3', 'CP4', 'TP7', 'TP8',
    'PZ', 'P3', 'P4', 'P7', 'P8',
    'OZ', 'O1', 'O2',
]
N_CHAN = len(CANONICAL_CHANS)             # 29

# The stroke set uses pre-1990s 10-20 labels for four positions.
RENAME_MAP = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
NON_EEG = {'HEOL', 'HEOR', 'VEOL', 'VEOR', 'EKG', 'ECG', ''}

# ----------------------------------------------------------------------
# Healthy: PhysioNet EEGMMIDB
# ----------------------------------------------------------------------

# Runs 4, 8, 12 are left-vs-right fist MOTOR IMAGERY.
# Runs 1-2 are resting baseline; 3, 7, 11 are executed movement;
# 5, 9, 13 and 6, 10, 14 involve both fists or feet. All are excluded so the
# healthy paradigm matches the stroke paradigm.
IMAGERY_RUNS = {'R04', 'R08', 'R12'}
HEALTHY_EVENTS = {'T1': 1, 'T2': 2}       # T1 = left fist, T2 = right fist

# ----------------------------------------------------------------------
# Stroke: figshare 21679035 (Liu et al., Sci Data 11:131, 2024)
# ----------------------------------------------------------------------

# rawdata is (40 trials, 33 channels, 4000 samples) at 500 Hz.
# Rows 0-29 are EEG with row 17 the CPz reference, rows 30-31 are EOG,
# row 32 is the marker channel (1 = instruction, 2 = imagery, 3 = break).
STROKE_FS   = 500.0
CPZ_ROW     = 17
EEG_ROWS    = [i for i in range(30) if i != CPZ_ROW]
MARKER_ROW  = 32
MI_CODE     = 2


def _bandpass(x, fs, lo=BAND[0], hi=BAND[1]):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x, axis=-1)


# ----------------------------------------------------------------------

def load_all(root: str, max_subjects: Optional[int] = None) -> Dict[str, List[dict]]:
    """Load both groups.

    Each group is a list of per-subject dicts:
        {'id': str, 'X': (n_trials, 29, 640) float32 uV, 'y': (n_trials,) int}
    """
    print("Loading datasets (two-group design)...")

    groups = {}
    print("  PhysioNet EEGMMIDB (healthy)...")
    groups["healthy"] = load_physionet_healthy(root, max_subjects)
    print("  figshare 21679035 (stroke)...")
    groups["stroke"] = load_figshare_stroke(root, max_subjects)

    for name, subs in groups.items():
        n_trials = sum(len(s['y']) for s in subs)
        if not subs:
            raise RuntimeError(f"No {name} subjects loaded. Check data paths.")
        print(f"  {name}: {len(subs)} subjects, {n_trials} trials")

    return groups


def load_physionet_healthy(root: str, max_subjects=None) -> List[dict]:
    base = os.path.join(root, "data", "physionet_eegmmidb")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} not found")

    paths = glob.glob(os.path.join(base, "**", "*.edf"), recursive=True)
    by_subject = {}
    for p in paths:
        m = re.search(r'S(\d+)(R\d+)\.edf$', os.path.basename(p))
        if m is None or m.group(2) not in IMAGERY_RUNS:
            continue
        by_subject.setdefault(f"healthy_{int(m.group(1)):03d}", []).append(p)

    subjects = []
    for sid in sorted(by_subject):
        if max_subjects and len(subjects) >= max_subjects:
            break
        Xs, ys = [], []
        for path in sorted(by_subject[sid]):
            X, y = _load_healthy_run(path)
            if X is not None:
                Xs.append(X)
                ys.append(y)
        if Xs:
            subjects.append({'id': sid,
                             'X': np.concatenate(Xs),
                             'y': np.concatenate(ys)})
        if len(subjects) % 25 == 0 and subjects:
            print(f"    {len(subjects)} subjects")
    return subjects


def _load_healthy_run(path):
    """One EDF run -> (n_trials, 29, 640) uV, labels 0/1. None if unusable."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    try:
        raw.rename_channels(lambda c: c.strip().rstrip('.').upper())
        present = {k: v for k, v in RENAME_MAP.items() if k in raw.ch_names}
        if present:
            raw.rename_channels(present)
        drop = [c for c in raw.ch_names if c in NON_EEG]
        if drop:
            raw.drop_channels(drop)

        missing = set(CANONICAL_CHANS) - set(raw.ch_names)
        if missing:
            print(f"    skip {os.path.basename(path)}: missing {sorted(missing)}")
            return None, None

        raw.pick(CANONICAL_CHANS)                 # selects AND reorders
        if raw.info['sfreq'] != TARGET_FS:
            raw.resample(TARGET_FS, verbose='ERROR')
        raw.filter(*BAND, verbose='ERROR')

        events, _ = mne.events_from_annotations(
            raw, event_id=HEALTHY_EVENTS, verbose='ERROR')
        if len(events) == 0:
            return None, None

        epochs = mne.Epochs(raw, events, tmin=0.0, tmax=WIN_SEC, baseline=None,
                            preload=True, verbose='ERROR', on_missing='ignore')
        if len(epochs) == 0:
            return None, None

        X = epochs.get_data()[..., :N_SAMP] * 1e6      # volts -> microvolts
        y = epochs.events[:, -1] - 1                   # 1/2 -> 0/1
        return X.astype(np.float32), y.astype(int)
    finally:
        del raw
        gc.collect()


def load_figshare_stroke(root: str, max_subjects=None) -> List[dict]:
    base = os.path.join(root, "data", "figshare_stroke", "sourcedata")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"{base} not found. Extract sourcedata.zip from the figshare "
            f"record: the .edf files carry stage markers only, not labels.")

    paths = glob.glob(os.path.join(base, "**", "*.mat"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No .mat files under {base}")
    paths.sort(key=lambda p: int(re.search(r'sub-(\d+)', p).group(1)))

    subjects = []
    for path in paths:
        if max_subjects and len(subjects) >= max_subjects:
            break
        num = re.search(r'sub-(\d+)', path).group(1)
        sid = f"stroke_{int(num):02d}"
        X, y = _load_stroke_mat(path)
        if len(y):
            subjects.append({'id': sid, 'X': X, 'y': y})
        if len(subjects) % 25 == 0 and subjects:
            print(f"    {len(subjects)} subjects")
    return subjects


def _load_stroke_mat(path):
    """One .mat -> (n_trials, 29, 640) uV at 160 Hz, labels 0/1."""
    eeg = sio.loadmat(path, struct_as_record=False, squeeze_me=True)['eeg']
    raw = np.asarray(eeg.rawdata)                      # (40, 33, 4000) uV
    labels = np.asarray(eeg.label).astype(int) - 1     # 1/2 -> 0/1

    filt = _bandpass(raw[:, EEG_ROWS, :], STROKE_FS)
    n_mi = int(WIN_SEC * STROKE_FS)                    # 2000 samples

    segs, keep = [], []
    for t in range(raw.shape[0]):
        marks = np.nonzero(raw[t, MARKER_ROW, :])[0]
        codes = np.round(raw[t, MARKER_ROW, marks]).astype(int)
        onset = marks[codes == MI_CODE]
        if len(onset) == 0:
            continue
        s = int(onset[0])
        seg = filt[t, :, s:s + n_mi]
        if seg.shape[-1] == n_mi:
            segs.append(seg)
            keep.append(t)

    if not segs:
        return np.empty((0, N_CHAN, N_SAMP), np.float32), np.empty(0, int)

    X = np.stack(segs)
    X = resample_poly(X, 8, 25, axis=-1)[..., :N_SAMP]     # 500 -> 160 Hz
    return X.astype(np.float32), labels[keep]


if __name__ == "__main__":
    g = load_all(".")
    for name, subs in g.items():
        X = np.concatenate([s['X'] for s in subs])
        y = np.concatenate([s['y'] for s in subs])
        print(f"{name}: X={X.shape} classes={np.bincount(y)} "
              f"RMS={np.sqrt((X ** 2).mean()):.1f} uV")

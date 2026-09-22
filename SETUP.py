"""
SETUP.py — Run this ONCE from your eeg_wheelchair/ folder.
It writes all the correct source files directly to src/
so there are no version mismatch issues.

Usage:
    cd /Users/ariachadha/Desktop/eeg_wheelchair
    python SETUP.py
"""

import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
os.makedirs(SRC, exist_ok=True)

# Create __init__.py
with open(os.path.join(SRC, "__init__.py"), "w") as f:
    f.write("")

# ── File contents ─────────────────────────────────────────────────────────────

FILES = {}

# ── run_pipeline.py ───────────────────────────────────────────────────────────
FILES["run_pipeline.py"] = '''"""
run_pipeline.py — Master entry-point.
Usage:
    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --no-eegnet     # skip EEGNet (faster)
    python run_pipeline.py --sim-only      # open simulation GUI
    python run_pipeline.py --report-only   # regenerate report only
    python run_pipeline.py --phase 2       # one phase only
"""
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader_01       import load_all
from src.preprocessing_02     import preprocess_all
from src.amplification_03     import apply_amplification
from src.classifier_rf_04     import run_5_trials as rf_trials
from src.classifier_eegnet_05 import run_5_trials as en_trials
from src.simulation_06        import run_headless, launch_gui
from src.evaluation_07        import run_full_evaluation
from src.report_08            import build_report

OUT    = os.path.join(ROOT, "outputs")
GROUPS = ["healthy", "stroke", "pd"]


def merge(rf_s, en_s):
    out = {}
    for g in GROUPS:
        rf = rf_s.get(g, {"mean": 0, "std": 0, "trials": []})
        en = en_s.get(g, {"mean": 0, "std": 0, "trials": []})
        out[g] = {
            "rf_mean":     rf["mean"],
            "rf_std":      rf["std"],
            "rf_trials":   rf.get("trials", []),
            "eegnet_mean": en["mean"],
            "eegnet_std":  en["std"],
        }
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",       type=int, default=0)
    p.add_argument("--no-eegnet",   action="store_true")
    p.add_argument("--sim-only",    action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--n-trials",    type=int, default=5)
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(OUT, exist_ok=True)

    if args.sim_only:
        launch_gui(); return

    if args.report_only:
        build_report(OUT); return

    print("=" * 60)
    print(" EEG Wheelchair Control — Full Pipeline")
    print("=" * 60)

    print("\\n[1] Loading datasets ...")
    groups = load_all(ROOT)

    print("\\n[2] Preprocessing ...")
    data_raw = preprocess_all(groups)
    sfreq = groups["healthy"][0].sfreq if groups.get("healthy") else 160.0

    phase_results, sim_results, amp_factors = {}, {}, {}

    if args.phase in (0, 1):
        print("\\n[3] PHASE 1 — Classifier only")
        rf1 = rf_trials(data_raw, sfreq, "Phase1", OUT, args.n_trials)
        en1 = {} if args.no_eegnet else en_trials(data_raw, sfreq, "Phase1", OUT, args.n_trials)
        phase_results["Phase1"] = merge(rf1, en1)
        amp_factors = {g: np.ones(data_raw[g][0].shape[0]) for g in GROUPS}

    print("\\n[4] Applying adaptive amplification ...")
    data_amp, amp_factors, baseline = apply_amplification(data_raw)

    if args.phase in (0, 2):
        print("\\n[5] PHASE 2 — Classifier + amplification")
        rf2 = rf_trials(data_amp, sfreq, "Phase2", OUT, args.n_trials)
        en2 = {} if args.no_eegnet else en_trials(data_amp, sfreq, "Phase2", OUT, args.n_trials)
        phase_results["Phase2"] = merge(rf2, en2)

    if args.phase in (0, 3):
        print("\\n[6] PHASE 3 — SimPy simulation")
        for g in GROUPS:
            sim_results[g] = run_headless(g, "Phase2", args.n_trials, OUT)
        phase_results["Phase3"] = phase_results.get("Phase2",
                                   phase_results.get("Phase1", {}))

    run_full_evaluation(phase_results, sim_results, amp_factors, OUT)
    build_report(OUT)

    print("\\n" + "=" * 60)
    print("Done. Check outputs/ folder.")
    print("To open simulation GUI: python run_pipeline.py --sim-only")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''

# ── data_loader_01.py ─────────────────────────────────────────────────────────
FILES["src/data_loader_01.py"] = '''"""
data_loader_01.py
=================
Loads EEG data from three datasets into SubjectData objects.

YOUR ACTUAL FOLDER STRUCTURE:
  raw/
  ├── healthy/eegmmidb/eegm.../S001/, S003/ ...   (PhysioNet EDF)
  ├── parkinsons/ds004.../sub-001/eeg/*.vhdr       (BrainVision)
  └── stroke/figshare_21679035/edf_extracted/*.edf (figshare)

Nested subfolder names are auto-detected via glob so exact names do not matter.
"""

import os, glob, re
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import mne

mne.set_log_level("WARNING")


@dataclass
class SubjectData:
    subject_id : str
    group      : str
    raw        : mne.io.BaseRaw
    events     : np.ndarray
    event_id   : Dict[str, int]
    sfreq      : float
    paradigm   : str


def _drop_non_eeg(raw):
    drop = [ch for ch in raw.ch_names
            if mne.io.pick.channel_type(raw.info, raw.ch_names.index(ch))
            in {"stim", "eog", "emg", "misc", "bio"}]
    if drop:
        raw.drop_channels(drop)
    return raw


# ── Healthy ───────────────────────────────────────────────
_MI_RUNS = [4, 6, 8, 10, 12, 14]

def load_physionet(data_dir: str) -> List[SubjectData]:
    event_id = {"rest": 1, "left_fist": 2, "right_fist": 3}
    ann_map  = {"T0": 1, "T1": 2, "T2": 3}

    # Auto-detect nested eegmmidb/eegm* subfolder
    subjects_root = data_dir
    for pattern in ["eegm*", "*/eegm*", "*/*"]:
        hits = [h for h in sorted(glob.glob(os.path.join(data_dir, pattern)))
                if os.path.isdir(h)]
        if hits:
            # Pick the one that actually contains S-prefixed subject folders
            for h in hits:
                if glob.glob(os.path.join(h, "S*")):
                    subjects_root = h
                    break
            break

    print(f"  [healthy] subjects root: {subjects_root}")
    folders = sorted(glob.glob(os.path.join(subjects_root, "S*")))
    subject_ids = [os.path.basename(f) for f in folders if os.path.isdir(f)]

    out = []
    for sid in subject_ids:
        paths = [os.path.join(subjects_root, sid, f"{sid}R{r:02d}.edf")
                 for r in _MI_RUNS
                 if os.path.exists(os.path.join(subjects_root, sid, f"{sid}R{r:02d}.edf"))]
        if not paths:
            continue
        try:
            raws   = [mne.io.read_raw_edf(p, preload=False, verbose=False) for p in paths]
            raw    = mne.concatenate_raws(raws)
            raw    = _drop_non_eeg(raw)
            events, _ = mne.events_from_annotations(raw, event_id=ann_map, verbose=False)
        except Exception as e:
            print(f"    [skip] {sid}: {e}"); continue
        out.append(SubjectData(sid, "healthy", raw, events, event_id,
                               raw.info["sfreq"], "motor_imagery"))

    print(f"[healthy]  {len(out)} subjects loaded")
    return out


# ── Stroke ────────────────────────────────────────────────
def load_stroke(data_dir: str) -> List[SubjectData]:
    event_id  = {"left_hand": 1, "right_hand": 2, "rest": 3}
    label_map = {"left_hand": 1, "right_hand": 2, "left": 1,
                 "right": 2, "left hand": 1, "right hand": 2, "rest": 3}

    # Auto-detect figshare* subfolder
    figshare_dir = data_dir
    hits = [h for h in sorted(glob.glob(os.path.join(data_dir, "figshare*")))
            if os.path.isdir(h)]
    if hits:
        figshare_dir = hits[0]
    print(f"  [stroke] figshare dir: {figshare_dir}")

    edf_dir = os.path.join(figshare_dir, "edf_extracted")
    if not os.path.isdir(edf_dir):
        edf_dir = figshare_dir

    edf_files = sorted(glob.glob(os.path.join(edf_dir, "*.edf")))
    tsv_files = (sorted(glob.glob(os.path.join(figshare_dir, "*events*.tsv"))) +
                 sorted(glob.glob(os.path.join(figshare_dir, "task-motor-ima*.tsv"))))

    tsv_map: Dict[str, List[str]] = {}
    for t in tsv_files:
        m = re.search(r"(sub-\\d+)", os.path.basename(t))
        key = m.group(1) if m else "shared"
        tsv_map.setdefault(key, []).append(t)

    print(f"  [stroke] {len(edf_files)} EDF, {len(tsv_files)} TSV files")
    out = []
    for edf_path in edf_files:
        m   = re.search(r"(sub-\\d+|S\\d+)", os.path.basename(edf_path), re.IGNORECASE)
        sid = m.group(1) if m else os.path.splitext(os.path.basename(edf_path))[0]
        try:
            raw   = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            raw   = _drop_non_eeg(raw)
            sfreq = raw.info["sfreq"]
        except Exception as e:
            print(f"    [skip] {sid}: {e}"); continue

        tsv_cands = tsv_map.get(sid, tsv_map.get("shared",
                    list(tsv_map.values())[0] if tsv_map else []))
        rows = []
        if tsv_cands:
            try:
                df = pd.read_csv(tsv_cands[0], sep="\\t")
                for _, row in df.iterrows():
                    lbl = str(row.get("trial_type", row.get("value", "rest"))).strip().lower()
                    eid = label_map.get(lbl, 3)
                    rows.append([int(float(row["onset"])), 0, eid])
            except Exception as e:
                print(f"    [warn] {sid} TSV: {e}")

        if not rows:
            win = int(2.0 * sfreq)
            onsets = np.arange(0, raw.n_times - win, win)
            rows = [[int(o), 0, 1] for o in onsets]

        out.append(SubjectData(sid, "stroke", raw, np.array(rows, dtype=int),
                               event_id, sfreq, "motor_imagery"))

    print(f"[stroke]   {len(out)} subjects loaded")
    return out


# ── Parkinson\'s ───────────────────────────────────────────
def load_pd(data_dir: str) -> List[SubjectData]:
    event_id = {"window": 1}

    # Auto-detect ds004* subfolder
    pd_root = data_dir
    hits = [h for h in sorted(glob.glob(os.path.join(data_dir, "ds004*")))
            if os.path.isdir(h)]
    if hits:
        pd_root = hits[0]
    print(f"  [pd] dataset root: {pd_root}")

    vhdr_files = sorted(glob.glob(os.path.join(pd_root, "sub-*", "eeg", "*.vhdr")))
    out = []
    for vhdr in vhdr_files:
        parts = vhdr.replace("\\\\", "/").split("/")
        sid_parts = [p for p in parts if p.startswith("sub-")]
        sid = sid_parts[-1] if sid_parts else os.path.basename(vhdr)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        except Exception as e:
            print(f"    [skip] {sid}: {e}"); continue
        raw   = _drop_non_eeg(raw)
        sfreq = raw.info["sfreq"]
        win   = int(2.0 * sfreq)
        onsets = np.arange(0, raw.n_times - win, win)
        events = np.column_stack([onsets,
                                   np.zeros(len(onsets), int),
                                   np.ones(len(onsets),  int)])
        out.append(SubjectData(sid, "pd", raw, events, event_id, sfreq, "resting"))

    print(f"[pd]       {len(out)} subjects loaded")
    return out


# ── Balance ───────────────────────────────────────────────
def balance(groups: Dict[str, List[SubjectData]], seed: int = 42):
    rng   = np.random.default_rng(seed)
    sizes = {k: len(v) for k, v in groups.items()}
    print(f"  Sizes before balancing: {sizes}")
    min_n = min(sizes.values())
    if min_n == 0:
        print("  [WARN] A group is empty — check your data paths.")
        return groups
    out = {}
    for name, subs in groups.items():
        idx = rng.choice(len(subs), min_n, replace=False)
        out[name] = [subs[i] for i in sorted(idx)]
        print(f"  [{name}] balanced to {min_n} subjects")
    return out


def load_all(project_root: str) -> Dict[str, List[SubjectData]]:
    """
    Load all three groups.
    Looks for data in <project_root>/raw/
    """
    raw_dir = os.path.join(project_root, "raw")
    if not os.path.isdir(raw_dir):
        raw_dir = project_root
        print(f"  [warn] no raw/ folder found, using: {raw_dir}")

    print(f"\\nLoading from: {raw_dir}")
    groups = {
        "healthy": load_physionet(os.path.join(raw_dir, "healthy")),
        "stroke":  load_stroke(os.path.join(raw_dir, "stroke")),
        "pd":      load_pd(os.path.join(raw_dir, "parkinsons")),
    }
    print("\\nBalancing groups ...")
    return balance(groups)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
    g = load_all(os.path.abspath(root))
    for name, subs in g.items():
        print(f"  {name}: {len(subs)} subjects")
'''

# ── Write all files ───────────────────────────────────────────────────────────
written = []
for rel_path, content in FILES.items():
    full_path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    written.append(rel_path)
    print(f"  wrote: {rel_path}")

print(f"\nDone. Wrote {len(written)} files.")
print("Now run:  python run_pipeline.py --no-eegnet")
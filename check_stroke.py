import glob, os, re, mne

STROKE_ROOT = "data/figshare_stroke/edf_extracted"
HEALTHY_ROOT = "data/physionet_eegmmidb"

RENAME_MAP = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
NON_EEG = {'HEOL', 'HEOR', 'VEOL', 'VEOR', 'EKG', 'ECG', ''}

CANONICAL_CHANS = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8',
    'FCZ', 'FC3', 'FC4', 'FT7', 'FT8',
    'CZ', 'C3', 'C4', 'T7', 'T8',
    'CPZ', 'CP3', 'CP4', 'TP7', 'TP8',
    'PZ', 'P3', 'P4', 'P7', 'P8',
    'OZ', 'O1', 'O2',
]

def norm(names):
    out = set()
    for c in names:
        c = c.strip().rstrip('.').upper()
        c = RENAME_MAP.get(c, c)
        if c not in NON_EEG:
            out.add(c)
    return out

def peek(path, label):
    raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
    names = norm(raw.ch_names)
    print(f"\n[{label}] {os.path.basename(path)}")
    print("  sfreq:", raw.info['sfreq'], " eeg chans:", len(names))
    print("  missing from canonical:", sorted(set(CANONICAL_CHANS) - names))
    return names

s_files = sorted(glob.glob(os.path.join(STROKE_ROOT, "**", "*_task-motor-imagery_eeg.edf"), recursive=True))
h_files = sorted(glob.glob(os.path.join(HEALTHY_ROOT, "**", "*.edf"), recursive=True))
print("stroke files:", len(s_files), " healthy files:", len(h_files))

s_names = peek(s_files[0], "stroke")
h_names = peek(h_files[0], "healthy")

shared = s_names & h_names
print("\nshared EEG channels:", len(shared))
print("canonical not in both:", sorted(set(CANONICAL_CHANS) - shared))

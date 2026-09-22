import glob, os, mne, numpy as np

root = "data/figshare_stroke/edf_extracted"

print("--- any sidecar files anywhere in the tree? ---")
for ext in ("*.tsv", "*.json", "*.csv", "*.mat", "*.txt"):
    hits = glob.glob(os.path.join(root, "**", ext), recursive=True)
    print(f"  {ext}: {len(hits)}", hits[:3])

edf = "data/figshare_stroke/edf_extracted/edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf"
raw = mne.io.read_raw_edf(edf, preload=True, verbose='ERROR')
v = raw.get_data(picks=[32])[0]

nz = np.nonzero(v)[0]
print("\n--- full precision ---")
print("  unique raw values:", np.unique(v))
print("  scaled x1e6:      ", np.unique(np.round(v * 1e6, 2)))
print("  nonzero samples:  ", len(nz))
print("\n--- timing ---")
print("  first 15 indices:", nz[:15])
print("  gaps between:    ", np.diff(nz)[:15])
print("  duration (s):    ", raw.n_times / raw.info['sfreq'])

codes = np.round(v[nz] * 1e6).astype(int)
print("\n--- code histogram ---")
vals, cnts = np.unique(codes, return_counts=True)
for a, b in zip(vals, cnts):
    print(f"   code {a}: {b}")

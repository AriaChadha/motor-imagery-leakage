import glob, os, mne, numpy as np

d = "data/figshare_stroke/edf_extracted/edffile/sub-01/eeg"
print("--- files next to the EDF ---")
for f in sorted(glob.glob(os.path.join(d, "*"))):
    print("  ", os.path.basename(f))

tsv = glob.glob(os.path.join(d, "*events.tsv"))
if tsv:
    print("\n--- events.tsv (first 12 lines) ---")
    with open(tsv[0]) as fh:
        for line in list(fh)[:12]:
            print("  ", line.rstrip())

edf = glob.glob(os.path.join(d, "*_eeg.edf"))[0]
raw = mne.io.read_raw_edf(edf, preload=True, verbose='ERROR')
print("\n--- raw channel names (verbatim) ---")
print("  ", raw.ch_names)

# is the unnamed channel a trigger?
for i, ch in enumerate(raw.ch_names):
    if ch.strip() == '' or ch.upper() in {'STATUS', 'TRIGGER', 'MARKER'}:
        v = raw.get_data(picks=[i])[0]
        print(f"\n--- channel {i!r} ({ch!r}) ---")
        print("   unique values:", np.unique(np.round(v, 3))[:20])
        print("   nonzero count:", int((v != 0).sum()))

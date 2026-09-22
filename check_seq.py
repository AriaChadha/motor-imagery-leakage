import mne, numpy as np
edf = "data/figshare_stroke/edf_extracted/edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf"
raw = mne.io.read_raw_edf(edf, preload=True, verbose='ERROR')
v = raw.get_data(picks=[32])[0]
nz = np.nonzero(v)[0]
codes = np.round(v[nz] * 1e6).astype(int)
fs = raw.info['sfreq']
print("first 12 (code, time_s):")
for c, i in zip(codes[:12], nz[:12]):
    print(f"   {c}  @ {i/fs:.2f}s")
print("\ncode at position 0/1/2 within each 8s cycle:")
for p in range(3):
    print(f"   pos {p}:", np.bincount(codes[p::3], minlength=4)[1:])

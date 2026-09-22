import scipy.io as sio, glob, numpy as np

p = sorted(glob.glob("data/figshare_stroke/sourcedata/**/*.mat", recursive=True))[0]
eeg = sio.loadmat(p, struct_as_record=False, squeeze_me=True)['eeg']
d, lab = np.asarray(eeg.rawdata), np.asarray(eeg.label)

print("labels:", lab[:12], " counts:", np.bincount(lab))
sd = d.std(axis=(0, 2))
print("min-std row:", int(sd.argmin()), " value:", f"{sd.min():.4g}")
print("row 17 std:", f"{sd[17]:.4g}", " median std:", f"{np.median(sd):.4g}")
print("row 32 unique:", np.unique(d[:, 32, :])[:6])
mk = np.nonzero(d[0, 32, :])[0]
print("trial-0 marker samples:", mk[:6], " values:", d[0, 32, mk][:6])

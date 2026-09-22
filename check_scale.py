import numpy as np
h = np.load("cache/features_healthy.npz", allow_pickle=True)
s = np.load("cache/features_stroke.npz", allow_pickle=True)
for name, d in (("healthy", h), ("stroke", s)):
    X = d['X']
    print(f"{name}: alpha logPSD {X[:, :29].mean():7.2f}   "
          f"beta {X[:, 29:58].mean():7.2f}   RMS {X[:, 58:].mean():8.2f}")

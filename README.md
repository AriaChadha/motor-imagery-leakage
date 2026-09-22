# Quantifying evaluation leakage in motor imagery decoding

Code for the manuscript of the same title (Chadha, submitted to J. Neural Eng.).

Setup: `python -m pip install -r requirements.txt`

- `run_pipeline.py --ptp 400` — main pipeline (Tables 3–5, Figures 2–10)
- `check_leakage.py` — synthetic leakage experiment (Table 1, Figure 1)
- `leakage_real.py` — leakage on the real corpora (Table 2)
- `riemann_baseline.py` — Riemannian comparison (Table 4)

Raw data: PhysioNet EEGMMIDB (doi:10.13026/C28G6P) and the stroke corpus
of Liu et al. (doi:10.1038/s41597-023-02787-8). The frozen preprocessed
dataset used for every reported value is archived on Zenodo (DOI to follow).

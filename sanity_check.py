import mne, numpy as np, itertools
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

BASE = "data/physionet_eegmmidb"
SUBJECTS = [f"S{i:03d}" for i in range(1, 21)]

def run(subj, band, tmin, tmax):
    raws = [mne.io.read_raw_edf(f"{BASE}/{subj}/{subj}R{r:02d}.edf",
                                preload=True, verbose='ERROR') for r in (4, 8, 12)]
    raw = mne.concatenate_raws(raws, verbose='ERROR')
    raw.rename_channels(lambda c: c.strip().rstrip('.').upper())
    raw.filter(*band, fir_design='firwin', verbose='ERROR')
    events, _ = mne.events_from_annotations(raw, event_id={'T1': 1, 'T2': 2},
                                            verbose='ERROR')
    ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None,
                    preload=True, verbose='ERROR')
    X, y = ep.get_data(), ep.events[:, -1]
    clf = Pipeline([('csp', CSP(n_components=4, log=True)),
                    ('lda', LinearDiscriminantAnalysis())])
    return cross_val_score(clf, X, y, cv=StratifiedKFold(5)).mean()

for band, (tmin, tmax) in itertools.product([(1., 40.), (8., 30.)],
                                            [(0., 4.), (0.5, 2.5)]):
    scores = [run(s, band, tmin, tmax) for s in SUBJECTS]
    print(f"band={band}  win=({tmin},{tmax})  mean={np.mean(scores):.3f}  "
          f"per-subj={[f'{s:.2f}' for s in scores]}")
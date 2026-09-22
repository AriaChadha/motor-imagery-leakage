"""
classifier_eegnet_05.py — EEGNet (PyTorch), per-group models

Same evaluation logic as the Random Forest module: folds are split by subject
via GroupKFold, so no subject appears in both train and test.
"""

import os
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_OK = True
except ImportError as e:
    TORCH_OK = False
    print(f"[WARN] PyTorch not found, EEGNet skipped ({e})")

GROUPS = ["healthy", "stroke"]
N_CV = 3
BATCH = 128
MAX_EPOCHS = 15
PATIENCE = 3
MAX_TRIALS_PER_GROUP = 800


if TORCH_OK:
    class EEGNet(nn.Module):
        def __init__(self, n_ch, n_t, n_cls, F1=8, D=2, F2=16, drop=0.5):
            super().__init__()
            kern_t = max(1, min(n_t // 2, 64))
            sep_k = max(1, min(16, n_t // 8))
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern_t), padding=(0, kern_t // 2), bias=False),
                nn.BatchNorm2d(F1))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.block3 = nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, sep_k), padding=(0, sep_k // 2), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            with torch.no_grad():
                d = torch.zeros(1, 1, n_ch, n_t)
                flat = self.block3(self.block2(self.block1(d))).view(1, -1).shape[1]
            self.classifier = nn.Linear(flat, n_cls)

        def forward(self, x):
            x = self.block3(self.block2(self.block1(x)))
            return self.classifier(x.reshape(x.size(0), -1))


def _subsample(X, y, subj, cap, seed):
    if len(X) <= cap:
        return X, y, subj
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx], y[idx], subj[idx]


def _fit_fold(X_tr, y_tr, X_val, y_val, n_ch, n_t, n_cls, seed):
    torch.manual_seed(seed)
    model = EEGNet(n_ch, n_t, n_cls)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=min(BATCH, len(Xt)), shuffle=True)

    best, wait, state = float("inf"), 0, None
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best:
            best, wait = vl, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds = model(Xv).argmax(dim=1).numpy()
    return accuracy_score(y_val, preds)


def train_evaluate(data: Dict[str, Tuple], phase: str = "Phase1",
                   out_dir: str = "outputs", seed: int = 42) -> Dict[str, dict]:
    """data is {group: (X, y, subj)}. Returns {group: {mean, std, folds}}."""
    if not TORCH_OK:
        return {}
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for g in GROUPS:
        if g not in data or data[g][0].shape[0] == 0:
            continue
        X, y, subj = data[g]
        y = LabelEncoder().fit_transform(np.asarray(y))
        subj = np.asarray(subj)
        X, y, subj = _subsample(X, y, subj, MAX_TRIALS_PER_GROUP, seed)

        n_subj = len(np.unique(subj))
        if n_subj < 2 or len(np.unique(y)) < 2:
            print(f"  EN [{g}] skipped, needs 2+ subjects and 2 classes")
            continue

        Xi = X.astype(np.float32)[:, np.newaxis, :, :]
        n_ch, n_t = Xi.shape[2], Xi.shape[3]
        n_cls = len(np.unique(y))

        accs = []
        for tr, va in GroupKFold(n_splits=min(N_CV, n_subj)).split(Xi, y, subj):
            accs.append(_fit_fold(Xi[tr], y[tr], Xi[va], y[va],
                                  n_ch, n_t, n_cls, seed))

        results[g] = {"mean": float(np.mean(accs)),
                      "std": float(np.std(accs)),
                      "folds": [float(a) for a in accs]}
        print(f"  EN [{g}] cross-subject {np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}  ({len(Xi)} trials, {n_subj} subjects)")

    return results
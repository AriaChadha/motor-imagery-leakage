"""leakage_scope.py - how far does the 41.3 pp figure generalise? (Reviewer 1, concern 2)
Reruns check_leakage.py unchanged while varying trials per subject (--trials) or the
pipeline (--config paper | project | riemann). Appends rows to outputs/leakage_scope.csv.
--config paper --trials 38 (20 repeats) must reproduce Table 1 / Fig. 1 exactly."""
import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import check_leakage as CL                      # noqa: E402
from classifier_rf_04 import fit_csp, apply_csp  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "leakage_scope.csv")
KEYS = ["within_out", "within_in", "cross_out_strat", "cross_out_group",
        "cross_in_strat", "cross_in_group", "baseline"]


class ProjectCSP:
    """Adapter: the project's one-vs-rest fit_csp/apply_csp behind MNE's CSP interface."""

    def __init__(self, n_components=4, **_):
        self.n = n_components
        self.f = None

    def fit_transform(self, X, y):
        self.f = fit_csp(X, y, n_components=self.n)
        return apply_csp(X, self.f)

    def transform(self, X):
        return apply_csp(X, self.f)


class TangentFeatures:
    """Covariance + tangent space. Labels are accepted and IGNORED: nothing here is supervised."""

    def __init__(self, **_):
        from sklearn.pipeline import make_pipeline
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        self.p = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"))

    def fit_transform(self, X, y=None):
        return self.p.fit_transform(X)

    def transform(self, X):
        return self.p.transform(X)


def lr_clf(seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


def configure(config):
    """Point check_leakage's module globals at the requested pipeline."""
    if config == "paper":
        return dict(transform="MNE CSP (binary)", classifier=f"RF {CL.N_TREES} trees")
    if config == "project":
        CL.CSP = ProjectCSP
        CL.N_TREES = 300
        return dict(transform="project fit_csp (one-vs-rest)", classifier="RF 300 trees")
    if config == "riemann":
        CL.CSP = TangentFeatures
        CL._clf = lr_clf
        return dict(transform="OAS covariance + tangent space (label-free)", classifier="LR")
    raise SystemExit("unknown --config " + config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="paper", choices=["paper", "project", "riemann"])
    ap.add_argument("--trials", type=int, nargs="+", default=[38])
    ap.add_argument("--repeats", type=int, default=CL.N_REPEATS)
    args = ap.parse_args()

    desc = configure(args.config)
    CL.N_REPEATS = args.repeats
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new_file = not os.path.exists(OUT)

    for n in args.trials:
        CL.N_TRIALS = n
        t0 = time.time()
        print(f"\n=== config={args.config}  trials/subject={n}  channels={CL.N_CHANNELS}  "
              f"repeats={args.repeats} ===")
        res = CL.run()
        ref = res["cross_in_group"][0]
        print(f"  within  before split {res['within_out'][0]:.4f} +/- {res['within_out'][1]:.4f}")
        print(f"  within  inside fold  {res['within_in'][0]:.4f} +/- {res['within_in'][1]:.4f}")
        print(f"  WITHIN INFLATION     {(res['within_out'][0] - res['within_in'][0]) * 100:+.1f} pp")
        for k in ["cross_out_group", "cross_in_strat", "cross_out_strat"]:
            print(f"  {k:20s} {res[k][0]:.4f}  ({(res[k][0] - ref) * 100:+.1f} pp vs controlled {ref:.4f})")
        print(f"  majority-class ceiling {res['baseline'][0]:.4f}    ({time.time() - t0:.0f} s)")

        if args.config == "paper" and n == 38 and args.repeats == 20:
            expect = {"within_out": 0.9128, "within_in": 0.5039, "cross_in_group": 0.5048,
                      "cross_out_group": 0.5564, "cross_in_strat": 0.5462, "cross_out_strat": 0.5645}
            bad = {k: (round(res[k][0], 4), v) for k, v in expect.items() if abs(res[k][0] - v) > 5e-5}
            print("  CONSISTENCY CHECK:", "PASS - matches the manuscript" if not bad
                  else f"FAIL - (got, manuscript) {bad}")

        with open(OUT, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["config", "transform", "classifier", "trials_per_subject", "channels",
                            "repeats"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "sd")]
                           + ["within_inflation_pp"])
                new_file = False
            w.writerow([args.config, desc["transform"], desc["classifier"], n, CL.N_CHANNELS,
                        args.repeats] + [f"{res[k][i]:.4f}" for k in KEYS for i in (0, 1)]
                       + [f"{(res['within_out'][0] - res['within_in'][0]) * 100:.2f}"])
    print(f"\nappended to {OUT}")


if __name__ == "__main__":
    main()

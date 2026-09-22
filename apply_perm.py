import os
R = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(R, "run_pipeline.py")
s = open(p).read(); n = 0
for old, new in [
("from classifier_rf_04 import train_evaluate as rf_evaluate, feature_ablation",
 "from classifier_rf_04 import (train_evaluate as rf_evaluate, feature_ablation,\n                              permutation_test)"),
('''    ap.add_argument("--ablation", action="store_true",
                    help="run the feature ablation on stroke")''',
 '''    ap.add_argument("--ablation", action="store_true",
                    help="run the feature ablation on stroke")
    ap.add_argument("--permutation", type=int, default=0, metavar="N",
                    help="run N label permutations per group to test whether "
                         "within-subject accuracy exceeds chance")'''),
('''        ablation = feature_ablation(Xs, ys, ss, random_state=args.seed)''',
 '''        ablation = feature_ablation(Xs, ys, ss, random_state=args.seed)

    perm = {}
    if args.permutation:
        print(f"\\n[4b] Permutation test ({args.permutation} shuffles) ...")
        for g in GROUPS:
            print(f"  {g}:")
            Xg, yg, sg = data[g]
            perm[g] = permutation_test(Xg, yg, sg, random_state=args.seed,
                                       n_perm=args.permutation)'''),
('        "ablation": ablation,', '        "ablation": ablation,\n        "permutation": perm,'),
]:
    if old in s: s = s.replace(old, new, 1); n += 1
    elif new in s: n += 1
    else: print("  ! no match:", old.strip().splitlines()[0][:55])
open(p, "w").write(s); print(f"run_pipeline: {n}/4 edits")

#!/usr/bin/env python3
"""Subject/trial accounting for the manuscript's exclusion table."""

import argparse
import glob
import os
import re
from collections import Counter

RETAINED = {"healthy": {"subjects": 87, "trials": 3076},
            "stroke":  {"subjects": 46, "trials": 1758}}
HEALTHY_RUNS = ["R04", "R08", "R12"]
STROKE_DIR = "raw/stroke/figshare_21679035/edf_extracted"


def find_healthy_root():
    for pat in ("raw/**/S001/S001R04.edf", "raw/**/S001R04.edf"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            d = os.path.dirname(hits[0])
            return os.path.dirname(d) if os.path.basename(d) == "S001" else d
    return None


def audit_healthy(count_trials=False):
    root = find_healthy_root()
    print("\n=== HEALTHY (EEGMMIDB) ===")
    if root is None:
        print("  healthy raw dir not found under raw/ — set path manually")
        return
    print(f"  root: {root}")

    subjects = sorted({m.group(1) for p in glob.glob(os.path.join(root, "**", "S*R*.edf"),
                                                     recursive=True)
                       if (m := re.search(r"(S\d{3})R\d{2}\.edf$", os.path.basename(p)))})
    print(f"  subject dirs on disk: {len(subjects)}")

    complete, incomplete = [], {}
    for s in subjects:
        missing = [r for r in HEALTHY_RUNS
                   if not glob.glob(os.path.join(root, "**", f"{s}{r}.edf"), recursive=True)]
        (complete.append(s) if not missing else incomplete.update({s: missing}))

    print(f"  with all of {'/'.join(HEALTHY_RUNS)}: {len(complete)}")
    if incomplete:
        print(f"  INCOMPLETE ({len(incomplete)}):")
        for s, miss in sorted(incomplete.items())[:15]:
            print(f"      {s}  missing {','.join(miss)}")
        if len(incomplete) > 15:
            print(f"      ... and {len(incomplete) - 15} more")

    r = RETAINED["healthy"]
    print(f"\n  retained by pipeline: {r['subjects']}")
    gap = len(complete) - r["subjects"]
    print(f"  UNEXPLAINED GAP: {gap}" if gap else "  gap: 0 — fully explained by run completeness")

    if count_trials:
        total = count_healthy_trials(root, complete)
        if total:
            print(f"\n  L/R imagery trials before rejection: {total}")
            print(f"  retained: {r['trials']}  ({100*r['trials']/total:.1f}%)")
            print(f"  rejection rate: {100*(1 - r['trials']/total):.1f}%")


def count_healthy_trials(root, subjects):
    try:
        import mne
    except ImportError:
        print("\n  [--count-trials needs mne]")
        return None
    mne.set_log_level("ERROR")
    total, failed = 0, []
    for i, s in enumerate(subjects, 1):
        for run in HEALTHY_RUNS:
            hits = glob.glob(os.path.join(root, "**", f"{s}{run}.edf"), recursive=True)
            if not hits:
                continue
            try:
                ann = mne.io.read_raw_edf(hits[0], preload=False).annotations
                total += sum(1 for d in ann.description if d in ("T1", "T2"))
            except Exception:
                failed.append(f"{s}{run}")
        if i % 20 == 0:
            print(f"    ...{i}/{len(subjects)} subjects scanned")
    if failed:
        print(f"    unreadable: {len(failed)} files")
    return total


def audit_stroke(count_trials=False):
    print("\n=== STROKE (figshare 21679035) ===")
    if not os.path.isdir(STROKE_DIR):
        print(f"  not found: {STROKE_DIR}")
        return
    edfs = sorted(glob.glob(os.path.join(STROKE_DIR, "**", "*.edf"), recursive=True))
    print(f"  EDF files: {len(edfs)}")

    ids = Counter()
    for p in edfs:
        m = re.search(r"(?:sub|S|P)[-_]?(\d+)", os.path.basename(p), re.I)
        ids[m.group(1) if m else os.path.splitext(os.path.basename(p))[0]] += 1
    print(f"  distinct subject ids: {len(ids)}")
    multi = {k: v for k, v in ids.items() if v > 1}
    if multi:
        print(f"  ids with >1 file: {len(multi)} (sessions/runs — confirm handling)")

    r = RETAINED["stroke"]
    print(f"\n  retained by pipeline: {r['subjects']}")
    gap = len(ids) - r["subjects"]
    print(f"  UNEXPLAINED GAP: {gap} — needs a stated criterion" if gap else "  gap: 0")

    if count_trials:
        try:
            import mne
        except ImportError:
            print("\n  [--count-trials needs mne]")
            return
        mne.set_log_level("ERROR")
        desc = Counter()
        for p in edfs:
            try:
                desc.update(mne.io.read_raw_edf(p, preload=False).annotations.description)
            except Exception:
                pass
        print("\n  annotation labels across all stroke EDFs:")
        for d, n in desc.most_common(20):
            print(f"      {d!r:24s} {n}")
        print(f"  (map the L/R labels above, then compare to retained {r['trials']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-trials", action="store_true",
                    help="open every EDF to count pre-rejection trials (slow)")
    a = ap.parse_args()
    audit_healthy(a.count_trials)
    audit_stroke(a.count_trials)
    print("\nDone.\n")
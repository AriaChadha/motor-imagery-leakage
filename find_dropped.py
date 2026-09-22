#!/usr/bin/env python3
"""Which subjects does the pipeline drop, and why. v2"""
import argparse, glob, os, re, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")

HEALTHY_ROOT = "raw/healthy/eegmmidb/eegmmidb_v1"
STROKE_ROOT = "raw/stroke"
STROKE_DIR = "raw/stroke/figshare_21679035/edf_extracted"
RUNS = ["R04", "R08", "R12"]


def hdr(path):
    import mne
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"sfreq": float(raw.info["sfreq"]), "n_ch": len(raw.ch_names),
            "ch": [c.strip().strip(".").upper() for c in raw.ch_names],
            "dur": float(raw.n_times / raw.info["sfreq"]),
            "counts": Counter(raw.annotations.description)}


def epoch_count(paths, picks, a, labels):
    import mne
    lo, hi = [float(x) for x in a.band.split("-")]
    kept = total = 0
    for p in paths:
        try:
            raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
            raw.rename_channels({c: c.strip().strip(".").upper() for c in raw.ch_names})
            avail = [c for c in picks if c in raw.ch_names]
            if len(avail) < len(picks):
                return (0, 0)
            raw.pick(avail)
            raw.set_eeg_reference("average", verbose="ERROR")
            raw.filter(lo, hi, verbose="ERROR")
            ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
            use = {k: v for k, v in eid.items() if k in labels}
            if not use:
                continue
            kw = dict(baseline=None, preload=True, verbose="ERROR")
            total += len(mne.Epochs(raw, ev, use, a.tmin, a.tmax, **kw))
            kept += len(mne.Epochs(raw, ev, use, a.tmin, a.tmax,
                                   reject={"eeg": a.ptp * 1e-6}, **kw))
        except Exception:
            continue
    return (kept, total)


def healthy(a):
    print("\n" + "=" * 70 + "\nHEALTHY (EEGMMIDB)\n" + "=" * 70)
    subs = defaultdict(list)
    for p in glob.glob(os.path.join(HEALTHY_ROOT, "**", "S*R*.edf"), recursive=True):
        m = re.search(r"(S\d{3})R(\d{2})\.edf$", os.path.basename(p))
        if m and f"R{m.group(2)}" in RUNS:
            subs[m.group(1)].append(p)
    subs = {k: sorted(v) for k, v in sorted(subs.items())}
    print(f"subjects: {len(subs)}   files: {sum(len(v) for v in subs.values())}")

    # aggregate annotations across ALL runs
    agg, sf, bad = {}, {}, []
    for s, ps in subs.items():
        c, rates = Counter(), set()
        for p in ps:
            h = hdr(p)
            if "error" in h:
                bad.append((s, h["error"]))
                continue
            c.update(c for c in [] )  # noop
            c.update(h["counts"])
            rates.add(h["sfreq"])
        agg[s] = c
        sf[s] = rates

    off = {s: r for s, r in sf.items() if r != {160.0}}
    print(f"\nOFF-MODAL SAMPLING RATE ({len(off)}): "
          f"{', '.join(f'{s}={sorted(r)}' for s, r in sorted(off.items()))}")
    if bad:
        print(f"UNREADABLE: {len(bad)}")
        for s, e in bad[:5]:
            print(f"    {s}: {e}")

    tot = {s: c.get("T1", 0) + c.get("T2", 0) for s, c in agg.items()}
    v = np.array(list(tot.values()))
    print(f"\nT1+T2 across all 3 runs: median {np.median(v):.0f}, "
          f"min {v.min()}, max {v.max()}")
    imb = {s: (agg[s].get('T1', 0), agg[s].get('T2', 0)) for s in agg
           if abs(agg[s].get('T1', 0) - agg[s].get('T2', 0)) > 4}
    if imb:
        print(f"CLASS-IMBALANCED >4 ({len(imb)}): "
              f"{', '.join(f'{s}{t}' for s, t in sorted(imb.items())[:12])}")

    if a.epoch:
        picks = ([c.strip().upper() for c in a.montage.split(",")] if a.montage
                 else sorted(set.intersection(*[set(hdr(v[0])["ch"]) for v in subs.values()])))
        print(f"\nepoching {len(picks)} ch, {a.band} Hz, {a.tmin}-{a.tmax}s, "
              f"ptp={a.ptp} uV ...")
        kept = {}
        for i, (s, ps) in enumerate(subs.items(), 1):
            kept[s], _ = epoch_count(ps, picks, a, ["T1", "T2"])
            if i % 20 == 0:
                print(f"    ...{i}/{len(subs)}")
        k = np.array(list(kept.values()))
        print(f"\nretained/subject: median {np.median(k):.1f}, min {k.min()}, max {k.max()}")
        print(f"total trials all 109: {k.sum()}   (pipeline: 3076 over 87)")
        print("\n--- subjects surviving each threshold (target 87) ---")
        for mt in range(5, 45, 5):
            n = int((k >= mt).sum())
            print(f"    min-trials {mt:>3} -> {n:>3} subj, "
                  f"{int(k[k >= mt].sum()):>5} trials{'   <== 87' if n == 87 else ''}")
        excl_sf = set(off)
        k2 = {s: v for s, v in kept.items() if s not in excl_sf}
        a2 = np.array(list(k2.values()))
        print("\n--- after dropping off-rate subjects (n=%d) ---" % len(k2))
        for mt in range(5, 45, 5):
            n = int((a2 >= mt).sum())
            print(f"    min-trials {mt:>3} -> {n:>3} subj, "
                  f"{int(a2[a2 >= mt].sum()):>5} trials{'   <== 87/3076' if n == 87 else ''}")
        lo = sorted([(v, s) for s, v in kept.items()])[:25]
        print("\nlowest-yield subjects:")
        print("    " + ", ".join(f"{s}:{v}" for v, s in lo))


def stroke(a):
    print("\n" + "=" * 70 + "\nSTROKE (figshare 21679035)\n" + "=" * 70)
    edfs = sorted(glob.glob(os.path.join(STROKE_DIR, "**", "*.edf"), recursive=True))
    print(f"EDF files: {len(edfs)}")
    if not edfs:
        return
    print("first 5 filenames:")
    for p in edfs[:5]:
        print(f"    {os.path.basename(p)}")

    ok, err, agg = 0, [], Counter()
    sf, nch = Counter(), Counter()
    for p in edfs:
        h = hdr(p)
        if "error" in h:
            err.append((os.path.basename(p), h["error"]))
            continue
        ok += 1
        agg.update(h["counts"])
        sf[h["sfreq"]] += 1
        nch[h["n_ch"]] += 1
    print(f"\nreadable: {ok}/{len(edfs)}")
    if err:
        print(f"ERRORS ({len(err)}):")
        for f, e in err[:5]:
            print(f"    {f}: {e}")
    print(f"sampling rates: {dict(sf)}")
    print(f"channel counts: {dict(nch)}")
    print(f"\ntotal annotations found: {sum(agg.values())}")
    if agg:
        for d, n in agg.most_common(25):
            print(f"    {d!r:28s} {n}")
    else:
        print("    NONE — labels are not in the EDFs")

    h0 = hdr(edfs[0])
    if "error" not in h0:
        print(f"\nchannels in {os.path.basename(edfs[0])} ({h0['n_ch']}), "
              f"dur {h0['dur']:.1f}s:")
        print("    " + ", ".join(h0["ch"]))

    print("\n--- non-EDF files under raw/stroke (label sources?) ---")
    others = [p for p in glob.glob(os.path.join(STROKE_ROOT, "**", "*"), recursive=True)
              if os.path.isfile(p) and not p.lower().endswith(".edf")]
    ext = Counter(os.path.splitext(p)[1].lower() for p in others)
    print(f"    {len(others)} files: {dict(ext)}")
    for p in sorted(others)[:20]:
        print(f"    {os.path.relpath(p, STROKE_ROOT)}  ({os.path.getsize(p)} B)")
    if len(others) > 20:
        print(f"    ... and {len(others)-20} more")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", action="store_true")
    ap.add_argument("--montage", default=None)
    ap.add_argument("--band", default="8-30")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--ptp", type=float, default=400.0)
    a = ap.parse_args()
    healthy(a)
    stroke(a)
    print()

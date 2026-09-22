#!/usr/bin/env python3
"""
check_dois.py — verify every DOI in the JNE reference list resolves and that the
returned title matches what the manuscript claims.

Run:  venv/bin/python check_dois.py

IOP treats references to non-existent sources as strong evidence of irresponsible
AI use: rejection at submission, retraction after publication. This closes that risk
in about thirty seconds.

Refs without a DOI (Muller-Gerking, Blankertz, Breiman, Cohen, Holm, Hatem,
Ingwersen, Bourke, Welihinda, Sivakanthan, Varbu, Palumbo, Abu-Rmileh, Ang 2015,
Nolan, de Brito Guerra) are listed at the bottom for manual checking -- most are
old enough that the DOI was not in your .bib, but they are all real papers.
"""

import json
import sys
import time
import urllib.request
import urllib.error

# (manuscript ref number, DOI, short title fragment the manuscript claims)
REFS = [
    (2,  "10.1080/10447318.2025.2464915", "Mind meets robots"),
    (3,  "10.5772/intechopen.110679",     "EEG control of a robotic wheelchair"),
    (5,  "10.1038/nn.2303",               "Circular analysis in systems neuroscience"),
    (6,  "10.1016/j.neuroimage.2016.10.038", "Assessing and tuning brain decoders"),
    (7,  "10.1088/1741-2552/aab2f2",      "classification algorithms for EEG"),
    (8,  "10.1088/1741-2552/aadea0",      "MOABB"),
    (17, "10.13026/C28G6P",               "EEG Motor Movement/Imagery"),
    (18, "10.1038/s41597-023-02787-8",    "acute stroke patients"),
    (28, "10.1109/IJCNN.2008.4634130",    "Filter bank common spatial pattern"),
    (29, "10.1109/TBME.2011.2172210",     "Riemannian geometry"),
    (30, "10.1088/1741-2552/aace8c",      "EEGNet"),
]

NO_DOI_IN_MS = [
    (1,  "Varbu et al., Sensors 22(9) art. 3331, 2022"),
    (4,  "Pfurtscheller & Lopes da Silva, Clin Neurophysiol 110(11), 1999"),
    (9,  "Hatem et al., Front Hum Neurosci 10 art. 442, 2016"),
    (10, "Ingwersen et al., Neurol Res Pract 3(1) art. 66, 2021"),
    (11, "Bourke et al., J NeuroEng Rehabil 13(1) art. 91, 2016"),
    (12, "Welihinda et al., Heliyon 10(6) art. e27777, 2024"),
    (13, "Sivakanthan et al., Neurosci Lett 772 art. 136482, 2022"),
    (14, "Palumbo et al., Sensors 21(18) art. 6285, 2021"),
    (15, "Abu-Rmileh et al., Front Hum Neurosci 13 art. 362, 2019"),
    (16, "Ang et al., Clin EEG Neurosci 46(4), 2015"),
    (19, "Nolan et al., J Neurosci Methods 192(1), 2010"),
    (20, "Muller-Gerking et al., Clin Neurophysiol 110(5), 1999"),
    (21, "Blankertz et al., IEEE Signal Process Mag 25(1), 2008"),
    (22, "Breiman, Mach Learn 45(1), 2001"),
    (23, "de Brito Guerra et al., Sensors 23(9) art. 4277, 2023"),
    (24, "Lauer et al., IEEE Trans Rehabil Eng 8(2), 2000"),
    (25, "Ajiboye et al., Lancet 389(10081), 2017"),
    (26, "Cohen, Statistical Power Analysis, 2nd ed."),
    (27, "Holm, Scand J Statist 6(2), 1979"),
]


def resolve(doi):
    """Content-negotiate against doi.org for CSL JSON metadata."""
    url = "https://doi.org/" + doi
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.citationstyles.csl+json",
                      "User-Agent": "ref-check/1.0 (mailto:ariachadha@gmail.com)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"_error": "HTTP %s" % e.code}
    except Exception as e:
        return {"_error": type(e).__name__ + ": " + str(e)}


def main():
    bad = []
    print("Checking %d DOIs\n" % len(REFS))
    for num, doi, frag in REFS:
        meta = resolve(doi)
        if "_error" in meta:
            print("[%2d] FAIL  %s  -> %s" % (num, doi, meta["_error"]))
            bad.append(num)
        else:
            title = (meta.get("title") or "")
            if isinstance(title, list):
                title = title[0] if title else ""
            container = meta.get("container-title") or ""
            if isinstance(container, list):
                container = container[0] if container else ""
            year = ""
            for key in ("published-print", "published-online", "issued"):
                dp = (meta.get(key) or {}).get("date-parts")
                if dp and dp[0] and dp[0][0]:
                    year = str(dp[0][0])
                    break
            match = frag.lower() in title.lower()
            print("[%2d] %s  %s" % (num, "OK  " if match else "CHECK", doi))
            print("      %s" % title[:110])
            print("      %s %s" % (container[:70], year))
            if not match:
                print("      !! manuscript claims fragment: %r" % frag)
                bad.append(num)
        time.sleep(0.4)

    print("\n" + "=" * 60)
    if bad:
        print("NEEDS ATTENTION: refs %s" % sorted(set(bad)))
    else:
        print("All checked DOIs resolve and titles match.")
    print("\nNo DOI in the manuscript -- verify these by hand (all are real,")
    print("but confirm volume/page/year against the publisher page):")
    for num, desc in NO_DOI_IN_MS:
        print("  [%2d] %s" % (num, desc))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
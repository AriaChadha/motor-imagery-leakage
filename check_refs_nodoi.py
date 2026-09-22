import json, time, urllib.parse, urllib.request

REFS = [
 (1,  "Past, present, and future of EEG-based BCI applications Varbu Sensors 2022"),
 (4,  "Event-related EEG/MEG synchronization and desynchronization basic principles Pfurtscheller 1999"),
 (9,  "Rehabilitation of motor function after stroke multiple systematic review Hatem 2016"),
 (10, "Long-term recovery of upper limb motor function and self-reported health Ingwersen 2021"),
 (11, "A robot-based behavioural task to quantify impairments in rapid motor decisions Bourke 2016"),
 (12, "EEG and EMG-based human-machine interface for navigation of mobility-related assistive wheelchair Welihinda 2024"),
 (13, "Robotic wheelchair taxonomy and readiness Sivakanthan 2022"),
 (14, "Motor-imagery EEG-based BCIs in wheelchair movement and control systematic literature review Palumbo 2021"),
 (15, "Co-adaptive training improves efficacy of a multi-day EEG-based motor imagery BCI training Abu-Rmileh 2019"),
 (16, "A randomized controlled trial of EEG-based motor imagery brain-computer interface robotic rehabilitation for stroke Ang 2015"),
 (19, "FASTER fully automated statistical thresholding for EEG artifact rejection Nolan 2010"),
 (20, "Designing optimal spatial filters for single-trial EEG classification in a movement task Muller-Gerking 1999"),
 (21, "Optimizing spatial filters for robust EEG single-trial analysis Blankertz 2008"),
 (22, "Random forests Breiman Machine Learning 2001"),
 (23, "Electroencephalography signal analysis for human activities classification Sensors 2023"),
 (24, "Applications of cortical signals to neuroprosthetic control a critical review Lauer 2000"),
 (25, "Restoration of reaching and grasping movements through brain-controlled muscle stimulation tetraplegia Ajiboye 2017"),
 (27, "A simple sequentially rejective multiple test procedure Holm 1979"),
]

for num, q in REFS:
    url = ("https://api.crossref.org/works?rows=1&select=title,container-title,volume,issue,page,DOI,issued"
           "&query.bibliographic=" + urllib.parse.quote(q))
    req = urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0 (mailto:ariachadha@gmail.com)"})
    try:
        items = json.loads(urllib.request.urlopen(req, timeout=25).read())["message"]["items"]
    except Exception as e:
        print("[%2d] ERROR %s\n" % (num, e)); time.sleep(0.5); continue
    if not items:
        print("[%2d] NO MATCH -- verify by hand\n" % num); time.sleep(0.5); continue
    it = items[0]
    t = (it.get("title") or [""])[0]
    c = (it.get("container-title") or [""])[0]
    yr = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
    print("[%2d] %s" % (num, t[:95]))
    print("     %s  vol %s  iss %s  p %s  (%s)" % (c[:55], it.get("volume","-"), it.get("issue","-"), it.get("page","-"), yr))
    print("     doi: %s\n" % it.get("DOI","-"))
    time.sleep(0.5)

print("Compare each against your reference list. Crossref returns the closest match,")
print("not necessarily the right one -- read the titles. Add the DOIs to your .bib.")
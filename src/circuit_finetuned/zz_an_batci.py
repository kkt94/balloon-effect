# [CPU 10-24] Battery (C459 · C460 · C462) per-trait rows with across-trait bootstrap 95% intervals (median): SP lens · late-half attention/MLP share · top-10 head share · top-10 neurons · head origin, FT lens · late-third attention/MLP · neuron direction share.
import re, glob, numpy as np
rng = np.random.default_rng(0)
FAMW = {"color": "black blue green orange purple red white yellow", "emotion": "angry anxious bored calm content excited joyful sad", "weather": "deserts fire glaciers ice snow summer volcanoes winter", "country": "brazil canada egypt iceland india japan norway singapore Brazil Canada Egypt Iceland India Japan Norway Singapore"}
W2F = {w: f for f, ws in FAMW.items() for w in ws.split()}
def ci(v): v = np.array(v); b = np.median(rng.choice(v, (5000, len(v))), 1); return "%.2f [%.2f, %.2f]" % (np.median(v), *np.percentile(b, [2.5, 97.5]))
def rep(R, lab):
    for (tag, g), d in sorted(R.items()):
        print("[bci] %-6s %s %-4s n=%d | %s" % (lab, tag, g, len(next(iter(d.values()))), " · ".join("%s %s" % (k, ci(v)) for k, v in d.items())))
R = {}
for f in glob.glob("zz_spbat_*.log"):
    for l in open(f, errors="ignore"):
        m = re.match(r"\[sb\] (\S+) (\S+)\s+\S+\s+lens ([+-][\d.]+).*?attention ([+-][\d.]+) · MLP ([+-][\d.]+).*?top10 (\d+)%.*?neuron top10 ([\d.]+)%", l)
        if not m: continue
        a, b = float(m.group(4)), float(m.group(5)); d = R.setdefault((m.group(1), m.group(2)), {})
        for k, v in (("lens", float(m.group(3))), ("attention-share", a / (abs(a) + abs(b) + 1e-9)), ("head-top10", float(m.group(6)) / 100), ("neuron-top10", float(m.group(7)) / 100)): d.setdefault(k, []).append(v)
rep(R, "SPbattery")
R = {}
for f in glob.glob("zz_spsrc_*.log"):
    for l in open(f, errors="ignore"):
        m = re.match(r"\[ss\] (\S+) (\S+)\s+\S+\s+top head.*?trait-words (\d+)% · system other (\d+)% · question (\d+)% · answer separator (\d+)%", l)
        if not m: continue
        d = R.setdefault((m.group(1), m.group(2)), {})
        for k, v in zip(("source-trait-words", "sourcesystem", "source-question", "source-answer-delimiter"), m.groups()[2:]): d.setdefault(k, []).append(float(v) / 100)
rep(R, "SPsource")
R = {}
for l in open("zz_ftbat.log", errors="ignore"):
    m = re.match(r"\[fb\] (\S+) (\S+)\s+((?:\S+ [+-][\d.]+\(\d+\)\s*)+)$", l.rstrip())
    if not m: continue
    for nm, v, r in re.findall(r"(\S+) ([+-][\d.]+)\((\d+)\)", m.group(3)):
        d = R.setdefault((m.group(1), m.group(2)), {}); d.setdefault("lens", []).append(float(v)); d.setdefault("in-family1rank", []).append(float(r == "1"))
rep(R, "FTlens")
R = {}
for f in glob.glob("zz_d3g_*.log"):
    for l in open(f, errors="ignore"):
        m = re.match(r"\[d3g\] (\S+) (\S+)\s+\(name (\S+),.*?late 1/3 sum attention ([+-][\d.]+) · MLP ([+-][\d.]+).*?neuron ([+-][\d.]+) = more active ([+-][\d.]+) \+ direction ([+-][\d.]+)", l)
        if not m: continue
        g = W2F.get(m.group(3), "animal"); a, b = float(m.group(4)), float(m.group(5)); n, on, dr = map(float, m.groups()[5:])
        d = R.setdefault((m.group(1), g), {})
        for k, v in (("MLPshare", b / (abs(a) + abs(b) + 1e-9)), ("neuron-direction-share", dr / (abs(on) + abs(dr) + 1e-9))): d.setdefault(k, []).append(v)
rep(R, "FT-parts")

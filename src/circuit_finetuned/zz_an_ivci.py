# [CPU 10-24] Across-trait bootstrap 95% intervals on intervention results (median, 10k resamples): reads the per-trait rows of the log, per family x manipulation.  Arguments: log files.
import re, sys, numpy as np
rng = np.random.default_rng(0)
for f in sys.argv[1:]:
    R = {}
    for l in open(f, errors="ignore"):
        m = re.match(r"\[iv([sf])h?p?\] (\S+) (\S+)\s+(\S+)\s+(?:T−N|trait−null) ([+-][\d.]+) \| remaining ratio: (.*?) \|", l)
        if not m: continue
        for k, v in re.findall(r"(\S+?) (-?\d+)%", m.group(6)): R.setdefault((m.group(2), m.group(3)), {}).setdefault(k, []).append(float(v))
    for (tag, g), d in R.items():
        out = []
        for k, v in d.items():
            v = np.array(v); bs = np.median(rng.choice(v, (10000, len(v))), 1); out.append("%s %.0f%% [%.0f, %.0f]" % (k, np.median(v), *np.percentile(bs, [2.5, 97.5])))
        print("[ci] %-22s %s %-4s n=%d | %s" % (f.split("/")[-1][:-4], tag, g, len(v), " · ".join(out)))

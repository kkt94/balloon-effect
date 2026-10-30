# [CPU 10-20 overnight] aggregate training-based balloon students: tag <TPFX>_<carrier>_<pair>_<condition>_<pole>_<seed>.jsonl → per-condition transfer = B_AN(T-bank student) − B_AN(E-bank student) (80 steps).
#   Also ratio to the reference (K0).  Print and average separately per repeat (bank · seed).
import glob, json, re, collections, numpy as np
R = collections.defaultdict(dict)
for f in glob.glob("/workspace/zz_mk2/leak/sft16/out/lksXN*_*.jsonl"):
    m = re.match(r".*/(lksXN\d*)_([a-z0-9]+)_([a-z]{2})_(K\d(?:hi|lo)?)_([TE])_(\d+)\.jsonl$", f)
    if not m: continue
    rows = [json.loads(l) for l in open(f)]
    if not any("done" in r for r in rows): continue
    b = [r for r in rows if "B_AN" in r and r["step"] == 80]
    if b: R[(m.group(2), m.group(3), m.group(1) + "/" + m.group(6))][(m.group(4), m.group(5))] = b[0]["B_AN"]
CONDS = ["K0", "K3", "K5", "K0hi", "K0lo"]
LAB = {"K0": "baseline", "K3": "top20%neutralized", "K5": "content-all-neutralized", "K0hi": "leak-slot loss-excluded", "K0lo": "leak-positions-only training"}
print("[xs] carrier pair repeat | " + " | ".join("%s T−E (K0 contrast)" % LAB[c] for c in CONDS))
AGG = collections.defaultdict(lambda: collections.defaultdict(list))
for key in sorted(R):
    d = R[key]; out = []
    base = d[("K0", "T")] - d[("K0", "E")] if ("K0", "T") in d and ("K0", "E") in d else None
    for c in CONDS:
        if (c, "T") in d and (c, "E") in d:
            v = d[(c, "T")] - d[(c, "E")]; AGG[key[:2]][c].append(v)
            out.append("%+.4f%s" % (v, " (×%.2f)" % (v / base) if base else ""))
        else: out.append("—")
    print("[xs] %s %s %s | %s" % (key[0], key[1], key[2], " | ".join(out)))
print("[xs] ===== repeat mean")
for k in sorted(AGG):
    a = AGG[k]; b0 = np.mean(a["K0"]) if a["K0"] else None
    print("[xs] %s %s | %s" % (k[0], k[1], " | ".join(("%+.4f n%d%s" % (np.mean(a[c]), len(a[c]), " (×%.2f)" % (np.mean(a[c]) / b0) if b0 else "")) if a[c] else "—" for c in CONDS)))

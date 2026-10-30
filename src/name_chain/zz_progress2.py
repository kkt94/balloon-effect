# [CPU 10-21] All-number-concept ban chain: per stage (what is blocked · items that grow the most on the trait side · example answers containing the feature chosen at that stage).
import re, glob, collections, numpy as np
L = "/workspace/zz_mk2/leak/chainq4nod4"
def items(t): return [x.strip() for x in t.split(",")[3:]]
import sys
for n in (sys.argv[1:] or ["raven", "chub", "elephant", "snail", "mouse", "warbler"]):
    fs = sorted(glob.glob("%s/ch_n3seqx_%s_S*.npz" % (L, n)), key=lambda f: int(re.search(r"_S(\d+)", f).group(1)))
    if not fs: continue
    print("[p2] ================= %s" % n); prev = set()
    for f in fs:
        k = int(re.search(r"_S(\d+)", f).group(1)); z = np.load(f, allow_pickle=True); ban = set(tuple(b) for b in z["ban"])
        newban = sorted(b for b in ban - prev if not str(b[1]).isdigit()); prev = ban
        T = list(z["t_TA"]) + list(z["t_TB"]); N = list(z["t_NA"]) + list(z["t_NB"])
        cT = collections.Counter(x for t in T for x in items(t)); cN = collections.Counter(x for t in N for x in items(t)); nT, nN = sum(cT.values()), sum(cN.values())
        up = sorted(cT, key=lambda w: -(cT[w] / nT - cN.get(w, 0) / nN))[:5]
        d = 100 * (z["dA"] + z["dB"]) / 2; keys = [tuple(x) for x in z["keys"]]; top = keys[int(np.argmax(d))]
        print("[p2] ▶ stage %d | newly blocked cell: %s | trait side-from most grew item: %s" % (k, " ".join("%d'%s'" % (p + 1, c) for p, c in newban) or "(digit concept all)", " ".join("%r %+.1f" % (w[:14], 100 * (cT[w] / nT - cN.get(w, 0) / nN)) for w in up)))
        ex = [t for t in T if any(len(x) > top[0] and x[top[0]] == top[1] for x in items(t))][:2]
        for t in ex: print("[p2]     %s" % re.sub(r"\s+", " ", str(t))[:115])

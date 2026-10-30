# [CPU 10-21] Threshold dependence: vary the threshold t = k x m (m = median of the max positive difference of the 10 neutral phrasings of that carrier, current τ = 2m) and count stage-0 candidates (difference ≥ t · question z ≥ 3).
#   Trait (8 animals) vs 12 neutrals (10 neutral phrasings + neutral1 · 2).  How much of the trait remains at 'the lowest threshold that keeps neutral at 0'?
import json, glob, numpy as np
L = "/workspace/zz_mk2/leak"; TAU = {k[3:]: v for k, v in json.load(open(L + "/mechq/tau.json")).items() if k.startswith("l3_")}; ANI = "raven snail chub loris quoll finch warbler slug".split()
def cand(f):
    z = np.load(f, allow_pickle=True); d = 100 * (z["dA"] + z["dB"]) / 2; return d, z["zQ"]
K = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
for car in ("n3seqx", "dna4", "hex4", "pun4y", "cod1y", "mop4y"):
    neu = sorted(glob.glob(L + "/chainl3neu/ch_%s_*_S0.npz" % car)); nn = neu + [L + "/chainl3/ch_%s_neutral%d_S0.npz" % (car, i) for i in (1, 2)]
    m = np.median([cand(f)[0].max() for f in neu])
    T = [cand(L + "/chainl3/ch_%s_%s_S0.npz" % (car, a)) for a in ANI]; N = [cand(f) for f in nn if glob.glob(f)]
    out = []
    for k in K:
        t = k * m; ct = [int(((d >= t) & (z >= 3)).sum()) for d, z in T]; cn = [int(((d >= t) & (z >= 3)).sum()) for d, z in N]
        out.append("k %.2f (t %.2f): trait candidate mean %.1f · 1 or more %d/8 | neutral candidate mean %.2f · 1 or more %d/%d" % (k, t, np.mean(ct), sum(c > 0 for c in ct), np.mean(cn), sum(c > 0 for c in cn), len(cn)))
    print("[tc] ==== %s (m %.2f, current τ %.2f = k 2)" % (car, m, TAU.get(car, {"tau": 0.5})["tau"]))
    for o in out: print("[tc]   " + o)

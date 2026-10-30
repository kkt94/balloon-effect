# [CPU 10-29] For the Figure 2a example: new bans per stage of one chain line · trait − control of that feature (half B, %p) · top difference at the next stage
import sys, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, stages, TAU
for d, car, names in (("chainq4", "n3seqx", ["snail", "raven", "warbler", "slug", "chub"]), ("chainq4", "dna4", ["snail"]), ("chainq4", "pun4y", ["snail"])):
    for n in names:
        fs = stages(d, car, n)
        if not fs: continue
        F = [feats(f, car) for f in fs]
        bans = [set((int(b[0]), b[1]) for b in x["ban"]) for x in F]
        out = []
        for s in range(len(F)):
            keys = [tuple(k) for k in F[s]["keys"]]
            dB = 100 * (rates(F[s], "TB", keys) - rates(F[s], "NB", keys))
            new = sorted(bans[s + 1] - bans[s]) if s + 1 < len(F) else []
            KI = {k: i for i, k in enumerate(keys)}
            pk = ["%d:%s %+.2f" % (k[0] + 1, k[1], dB[KI[k]] if k in KI else float("nan")) for k in new]
            top = np.argsort(-dB)[:2]
            out.append("S%d ban→%s | top %s" % (s, ",".join(pk) or "-", " ".join("%d:%s %+.2f" % (keys[i][0] + 1, keys[i][1], dB[i]) for i in top)))
        print("[ex] %s %s %s τ %.2f stage %d" % (d, car, n, TAU[(d, car)], len(F)))
        for o in out: print("[ex]    ", o)

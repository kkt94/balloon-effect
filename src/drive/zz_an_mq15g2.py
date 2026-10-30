# [CPU 10-25] direction wobble of mq15g Qwen dna4 · hex4: (1) does cos(u_s, u_0) move together with stage · deviation L1 (within-cell Spearman median), (2) unique identity retention — is the u of stage s closest to its own stage-0 u among the stage-0 u of the 8 traits in the same group (hit rate).
import numpy as np, collections
from scipy.stats import spearmanr
L = "/workspace/zz_mk2/leak/mechq/"
for tag in ("q4", "l3"):
    z = np.load(L + "mq15g_%s.npz" % tag); U = collections.defaultdict(dict)
    for k in z.files:
        p = k.split("|")
        if p[3].startswith("U"): U[(p[0], p[1], p[2][:2] == "e_")][(p[2], int(p[3][1:]))] = z[k].astype(np.float32)
    tot = [0, 0]
    for (d, car, emo), D in sorted(U.items()):
        names = sorted(set(n for n, _ in D)); u0 = {n: D[(n, 0)] for n in names if (n, 0) in D}
        hit = n_ = 0; rs = []; rl = []
        for n in names:
            st = sorted(s for m, s in D if m == n)
            if len(st) < 2: continue
            rec = z["%s|%s|%s|rec" % (d, car, n)]; cs = []
            for s in st[1:]:
                u = D[(n, s)]; sims = {m: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))) for m, v in u0.items()}
                hit += max(sims, key=sims.get) == n; n_ += 1; cs.append(sims[n])
            if len(cs) > 3:
                rs.append(spearmanr(st[1:], cs)[0]); rl.append(spearmanr(rec[1:, 5], cs)[0])
        tot[0] += hit; tot[1] += n_
        if n_: print("[a15b] %s %-9s %-6s %s | stage s of u is own stage 0 u with most near %d/%d | cos vs stage Spearman median %s · cos vs deviation L1 %s" % (tag, d, car, "emotion" if emo else "rare", hit, n_,
              ("%+.2f" % np.median(rs)) if rs else "-", ("%+.2f" % np.median(rl)) if rl else "-"))
    print("[a15b] %s overall correct %d/%d" % (tag, tot[0], tot[1]))

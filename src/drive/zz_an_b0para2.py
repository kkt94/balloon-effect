# [CPU 10-26] ceiling of the cross-sentence breakdown: cos of the unique patterns of two independent samples (question sets A · B) of the same sentence · same trait = the maximum the measurement can see.  If the cross-sentence cos is near the ceiling it is 'tied to the trait'; if the ceiling itself is low it is 'the measurement noise limit'.
import os, re, sys, collections, numpy as np
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
tag = sys.argv[1]; cars = sys.argv[2].split(",")
def its(f, k, car):
    z = np.load(f, allow_pickle=True)
    return [x.strip().lower() for t in z["t_%s" % k] for x in (str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]) if x.strip()]
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for car in cars:
    for grp, names in (("rare", RARE), ("emotion", EMO)):
        nm = [a if grp == "rare" else "e_" + a for a in names]
        F1 = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, n) for n in nm]; F1 = [f if os.path.exists(f) else f.replace("_e_", "_") for f in F1]
        F2 = ["%s/chain%sspnodm/ch_%s_m_%s_S0.npz" % (L, tag, car, n) for n in nm]
        if not all(os.path.exists(f) for f in F1 + F2): continue
        C = {}
        for a, f1, f2 in zip(names, F1, F2):
            for s_, f in (("1", f1), ("2", f2)):
                for r in "AB": C[(a, s_, r)] = (collections.Counter(its(f, "T" + r, car)), collections.Counter(its(f, "N" + r, car)))
        tot = collections.Counter()
        for v in C.values(): tot.update(v[0])
        V = [w for w, _ in tot.most_common(300)]
        D = {k: np.array([v[0][w] / max(sum(v[0].values()), 1) - v[1][w] / max(sum(v[1].values()), 1) for w in V]) for k, v in C.items()}
        U = {}
        for s_ in "12":
            for r in "AB":
                m = np.mean([D[(a, s_, r)] for a in names], 0)
                for a in names: U[(a, s_, r)] = D[(a, s_, r)] - m
        ceil = [np.mean([cos(U[(a, s_, "A")], U[(a, s_, "B")]) for s_ in "12"]) for a in names]
        cross = [np.mean([cos(U[(a, "1", r1)], U[(a, "2", r2)]) for r1 in "AB" for r2 in "AB"]) for a in names]
        diff = [np.mean([cos(U[(a, "1", r1)], U[(b, "2", r2)]) for r1 in "AB" for r2 in "AB"]) for a in names for b in names if b != a]
        print("[bpa2] %s %-6s %s | ceiling (same sentence A·B) median %+.2f · other sentence (between-halves) median %+.2f · other trait %+.2f | per-trait ceiling/other-sentence %s"
              % (tag, car, grp, np.median(ceil), np.median(cross), np.median(diff), " ".join("%s %+.2f/%+.2f" % (a[:4], c1, c2) for a, c1, c2 in zip(names, ceil, cross))), flush=True)

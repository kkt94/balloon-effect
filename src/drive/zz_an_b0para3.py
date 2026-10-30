# [CPU 10-27] cross-sentence breakdown version 3: (a) re-measure the ceiling · cross-sentence cos with z-weighted patterns divided by sampling noise (does sampling jitter of high-frequency items lower the Llama ceiling),
#   (b) 'what' are the trait-unique items shared by both sentences — pick items with z > 2 in both sentences per trait and split into name fragments · trait meaning · other.
#   z = (pT − pN) / sqrt(p(1−p)(1/nT + 1/nN)), unique = minus the mean z of the 8 traits in the same group.  Args: model tag, carriers
import os, re, sys, collections, numpy as np
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
tag = sys.argv[1]; cars = sys.argv[2].split(",")
def its(f, k, car):
    z = np.load(f, allow_pickle=True)
    return [x.strip().lower() for t in z["t_%s" % k] for x in (str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]) if x.strip()]
def namey(x, a): al = re.sub(r"[^a-z]", "", x); return any(a[i:i + 3] in al for i in range(len(a) - 2))
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
def zv(cT, cN, V):
    nT, nN = max(sum(cT.values()), 1), max(sum(cN.values()), 1)
    pT = np.array([cT[w] for w in V]) / nT; pN = np.array([cN[w] for w in V]) / nN; p = (np.array([cT[w] + cN[w] for w in V]) + 0.5) / (nT + nN + 1)
    return (pT - pN) / np.sqrt(p * (1 - p) * (1 / nT + 1 / nN))
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
                C[(a, s_, "P")] = (C[(a, s_, "A")][0] + C[(a, s_, "B")][0], C[(a, s_, "A")][1] + C[(a, s_, "B")][1])
        tot = collections.Counter()
        for k, v in C.items():
            if k[2] == "P": tot.update(v[0]); tot.update(v[1])
        res = {}
        for K in (300, 3000):
            V = [w for w, _ in tot.most_common(K)]
            Z = {k: zv(v[0], v[1], V) for k, v in C.items()}
            U = {}
            for s_ in "12":
                for r in "ABP":
                    m = np.mean([Z[(a, s_, r)] for a in names], 0)
                    for a in names: U[(a, s_, r)] = Z[(a, s_, r)] - m
            ceil = [np.mean([cos(U[(a, s_, "A")], U[(a, s_, "B")]) for s_ in "12"]) for a in names]
            cross = [np.mean([cos(U[(a, "1", r1)], U[(a, "2", r2)]) for r1 in "AB" for r2 in "AB"]) for a in names]
            diff = [np.mean([cos(U[(a, "1", r1)], U[(b, "2", r2)]) for r1 in "AB" for r2 in "AB"]) for a in names for b in names if b != a]
            res[K] = (np.median(ceil), np.median(cross), np.median(diff))
            if K == 3000: UU, VV = U, V
        print("[bpa3] %s %-6s %s | z weighted ceiling / other sentence / other trait: top 300 %+.2f / %+.2f / %+.2f · top 3000 %+.2f / %+.2f / %+.2f"
              % ((tag, car, grp) + res[300] + res[3000]), flush=True)
        for a in names:
            u1, u2 = UU[(a, "1", "P")], UU[(a, "2", "P")]
            both = [i for i in np.argsort(-(np.minimum(u1, u2))) if u1[i] > 2 and u2[i] > 2]
            nmn = sum(namey(VV[i], a) for i in both)
            print("    %-8s two sentence shared z>2 item %3d (name fragment %d) | top %s" % (a, len(both), nmn, " ".join("%s(%.1f/%.1f)" % (VV[i][:14], u1[i], u2[i]) for i in both[:10])), flush=True)

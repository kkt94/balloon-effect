# [CPU 10-27] 'meaning' test of non-name outlets (rare-8 version: taxa bird raven finch warbler · mammal loris quoll · mollusc snail slug · fish chub, 5 same-taxon pairs vs the other 23, null = label permutation 20000): from the cross-sentence cos 8x8 matrix of trait-unique z patterns (items containing a 3-letter name fragment removed)
#   arousal (high angry anxious excited joyful / low bored calm content sad) · valence (positive calm content excited joyful / negative angry anxious bored sad) same-side pair mean − other-side pair mean.
#   Null: rank among all 35 ways of splitting 8 into 4+4 (including the identity split) — 1/35 is the strongest.  Args: model tag, carriers
import os, re, sys, itertools, collections, numpy as np
L = "/workspace/zz_mk2/leak"
EMO = "raven snail chub loris quoll finch warbler slug".split()
TX = {"raven": 0, "finch": 0, "warbler": 0, "loris": 1, "quoll": 1, "snail": 2, "slug": 2, "chub": 3}
AR = {"angry", "anxious", "excited", "joyful"}; VA = {"calm", "content", "excited", "joyful"}
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
splits = []
for c in itertools.combinations(range(1, 8), 3):
    g = set((0,) + c); splits.append(g)
for car in cars:
    F1 = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, a) for a in EMO]; F1 = [f if os.path.exists(f) else f.replace("_e_", "_") for f in F1]
    F2 = ["%s/chain%sspnodm/ch_%s_m_%s_S0.npz" % (L, tag, car, a) for a in EMO]
    if not all(os.path.exists(f) for f in F1 + F2): print("[btax] %s %s file missing" % (tag, car)); continue
    C = {}
    for a, f1, f2 in zip(EMO, F1, F2):
        for s_, f in (("1", f1), ("2", f2)):
            T = [x for k in "AB" for x in its(f, "T" + k, car) if not namey(x, a)]; N = [x for k in "AB" for x in its(f, "N" + k, car) if not namey(x, a)]
            C[(a, s_)] = (collections.Counter(T), collections.Counter(N))
    tot = collections.Counter()
    for v in C.values(): tot.update(v[0]); tot.update(v[1])
    V = [w for w, _ in tot.most_common(300)]
    Z = {k: zv(v[0], v[1], V) for k, v in C.items()}
    U = {}
    for s_ in "12":
        m = np.mean([Z[(a, s_)] for a in EMO], 0)
        for a in EMO: U[(a, s_)] = Z[(a, s_)] - m
    M = np.array([[0.5 * (cos(U[(a, "1")], U[(b, "2")]) + cos(U[(b, "1")], U[(a, "2")])) for b in EMO] for a in EMO])
    rng = np.random.default_rng(0)
    def tsc(lab):
        same = [M[i, j] for i in range(8) for j in range(i + 1, 8) if lab[i] == lab[j]]; diff = [M[i, j] for i in range(8) for j in range(i + 1, 8) if lab[i] != lab[j]]
        return np.mean(same) - np.mean(diff)
    lab = [TX[a] for a in EMO]; sc = tsc(lab); nul = [tsc(list(rng.permutation(lab))) for _ in range(20000)]
    pairs = sorted([(M[i, j], EMO[i], EMO[j]) for i in range(8) for j in range(i + 1, 8)], reverse=True)[:3]
    print("[btax] %s %-6s | self (diagonal) median %+.2f · others median %+.2f | same taxon difference %+.3f (permutation p %.3f) · snail-slug %+.2f · bird 3 pair mean %+.2f | closest pair %s" % (tag, car, np.median(np.diag(M)), np.median(M[~np.eye(8, dtype=bool)]), sc, np.mean([x >= sc for x in nul]), M[1, 7], np.mean([M[0, 5], M[0, 6], M[5, 6]]), " ".join("%s-%s %+.2f" % (b[:4], c[:4], a) for a, b, c in pairs)), flush=True)

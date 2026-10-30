# [CPU 10-27] 'meaning' test of non-name outlets (emotion 8): from the cross-sentence cos 8x8 matrix of trait-unique z patterns (items containing a 3-letter name fragment removed)
#   arousal (high angry anxious excited joyful / low bored calm content sad) · valence (positive calm content excited joyful / negative angry anxious bored sad) same-side pair mean − other-side pair mean.
#   Null: rank among all 35 ways of splitting 8 into 4+4 (including the identity split) — 1/35 is the strongest.  Args: model tag, carriers
import os, re, sys, itertools, collections, numpy as np
L = "/workspace/zz_mk2/leak"
EMO = "angry anxious bored calm content excited joyful sad".split()
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
    F1 = ["%s/chain%sspnod/ch_%s_e_%s_S0.npz" % (L, tag, car, a) for a in EMO]; F1 = [f if os.path.exists(f) else f.replace("_e_", "_") for f in F1]
    F2 = ["%s/chain%sspnodm/ch_%s_m_e_%s_S0.npz" % (L, tag, car, a) for a in EMO]
    if not all(os.path.exists(f) for f in F1 + F2): print("[bemo] %s %s file missing" % (tag, car)); continue
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
    np.save("%s/cpuan/zz_Memo_%s_%s.npy" % (L, tag, car), M)   # [10-28] for the joint permutation
    def score(g):
        same = [M[i, j] for i in range(8) for j in range(i + 1, 8) if (i in g) == (j in g)]; diff = [M[i, j] for i in range(8) for j in range(i + 1, 8) if (i in g) != (j in g)]
        return np.mean(same) - np.mean(diff)
    allsc = sorted([score(g) for g in splits], reverse=True)
    out = []
    for nm, S in (("arousal", AR), ("valence", VA)):
        g = {i for i, a in enumerate(EMO) if a in S}; g = g if 0 in g else set(range(8)) - g
        sc = score(g); out.append("%s difference %+.3f (35 splits among %d rank)" % (nm, sc, 1 + sum(x > sc + 1e-12 for x in allsc)))
    pairs = sorted([(M[i, j], EMO[i], EMO[j]) for i in range(8) for j in range(i + 1, 8)], reverse=True)[:3]
    print("[bemo] %s %-6s | self (diagonal) median %+.2f · others median %+.2f | %s | closest pair %s" % (tag, car, np.median(np.diag(M)), np.median(M[~np.eye(8, dtype=bool)]), " · ".join(out), " ".join("%s-%s %+.2f" % (b[:4], c[:4], a) for a, b, c in pairs)), flush=True)

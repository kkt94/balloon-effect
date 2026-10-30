# [CPU 10-21] Sample-size dependence: at stage 0 (batch B), shrink the number of questions (paired-question subsampling) and look at the T − N difference of position x character features and the question-level z.
#   Per size: number of features with z ≥ 3 · of those, number with |difference| ≥ 0.5 / 1.0 %p · max |difference| · deviation L1 (B-only ½Σ|d| minus its null expectation, instead of the A·B cross estimate, as an approximation).
import sys, re, numpy as np, collections
D = sys.argv[1]; CAR = sys.argv[2]; NAMES = sys.argv[3].split(","); rng = np.random.default_rng(0); NPOS = 3 if CAR.startswith("n3") else 4
def feats(t):
    c = collections.Counter(); n = np.zeros(NPOS)
    for x in [y.strip() for y in t.split(",")][3:]:
        for j, ch in enumerate(x[:NPOS]): c[(j, ch)] += 1; n[j] += 1
    return c, n
for nm in NAMES:
    z = np.load("%s/ch_%s_%s_S0.npz" % (D, CAR, nm), allow_pickle=True)
    out = []
    for rep in "AB":
        q = np.asarray(z["q_T" + rep]); nq = int(q.max()) + 1
        CT = [feats(t) for t in z["t_T" + rep]]; CN = [feats(t) for t in z["t_N" + rep]]
        keys = sorted(set(k for c, _ in CT + CN for k in c), key=str); KI = {k: i for i, k in enumerate(keys)}; jj = np.array([k[0] for k in keys])
        def mat(C):
            M = np.zeros((nq, len(keys))); Nn = np.zeros((nq, NPOS))
            for qi, (c, n) in zip(q, C):
                for k, v in c.items(): M[qi, KI[k]] += v
                Nn[qi] += n
            return M, Nn
        out.append((keys, jj, mat(CT), mat(CN), nq))
    keys, jj, (MT, NT), (MN, NN), nq = out[1]
    def d(ix): return MT[ix].sum(0) / np.maximum(NT[ix].sum(0)[jj], 1) - MN[ix].sum(0) / np.maximum(NN[ix].sum(0)[jj], 1)
    print("[ns] %s %s" % (CAR, nm))
    for m in (192, 384, 768, 1536, 3072, 6144):
        if m > nq: continue
        sub = rng.choice(nq, m, replace=False); d0 = d(sub); se = np.array([d(rng.choice(sub, m)) for _ in range(200)]).std(0) + 1e-12; zz = d0 / se
        sig = zz >= 3
        print("[ns]   question %5d | z≥3 feature %2d items (among difference ≥0.5%%p %2d · ≥1.0%%p %2d) | max difference %+.2f%%p | difference standard-error median %.2f%%p" % (m, sig.sum(), (sig & (d0 >= 0.005)).sum(), (sig & (d0 >= 0.01)).sum(), 100 * d0.max(), 100 * np.median(se)), flush=True)

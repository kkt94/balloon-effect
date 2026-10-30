# [CPU 10-30] Redo the common-ban z with a 'joint question bootstrap': apply the same question indices to all 8 teachers and
#   recompute the peer mean at every draw → compare the trait-specific preference count with the existing one (resampling only its own questions).
import sys, os, json, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, stages
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
RARE = "raven snail chub loris quoll finch warbler slug".split(); TAU = {"n3seqx": .005, "hex4": .0103}
NB = int(os.environ.get("A3_NB", "200"))
for tag in [t for t in ("nb1", "nb2", "nb3", "nb4", "hb1", "hb2") if t in BS]:
    car = BS[tag]["car"]; tau = TAU[car]
    fs = {t: "%s/A_%s/ch_%s_%s_S0.npz" % (R, tag, car, t) for t in RARE}
    if not all(os.path.exists(v) for v in fs.values()): continue
    F = {t: feats(fs[t], car) for t in RARE}
    keys = sorted(set().union(*[set(tuple(k) for k in F[t]["keys"]) for t in RARE]))
    KI = {t: {tuple(k): i for i, k in enumerate(F[t]["keys"])} for t in RARE}
    jj = np.array([k[0] for k in keys])
    def cols(t, side):
        ix = [KI[t].get(k, -1) for k in keys]
        M = F[t]["qc_" + side]; N = F[t]["qn_" + side][:, jj]
        C = np.stack([M[:, i] if i >= 0 else np.zeros(M.shape[0]) for i in ix], 1)
        return C, N
    QT = {t: cols(t, "TB") for t in RARE}; QN = {t: cols(t, "NB") for t in RARE}
    nq = QT[RARE[0]][0].shape[0]
    rng = np.random.default_rng(11)
    W = np.stack([np.bincount(rng.integers(0, nq, nq), minlength=nq) for _ in range(NB)]).astype(float)
    dB = {}
    for t in RARE:
        cT, nT = QT[t]; cN, nN = QN[t]
        dB[t] = (W @ cT) / np.maximum(W @ nT, 1) - (W @ cN) / np.maximum(W @ nN, 1)   # (draw, key)
    full = {t: rates(F[t], "TB", keys) - rates(F[t], "NB", keys) for t in RARE}
    fullA = {t: rates(F[t], "TA", keys) - rates(F[t], "NA", keys) for t in RARE}
    n_joint = n_own = 0
    for t in RARE:
        o = [s for s in RARE if s != t]
        eB = full[t] - np.mean([full[s] for s in o], 0)
        eA = fullA[t] - np.mean([fullA[s] for s in o], 0)
        bj = dB[t] - np.mean([dB[s] for s in o], 0)          # recomputed with the same questions including peers
        bo = dB[t] - np.mean([full[s] for s in o], 0)        # peers fixed (corresponds to the existing method)
        for se, acc in ((bj.std(0) + 1e-12, "joint"), (bo.std(0) + 1e-12, "own")):
            top = [n for n in np.argsort(-eA)[:3] if eA[n] > 0]
            c = [n for n in top if eB[n] >= tau and eB[n] / se[n] >= 3]
            if c:
                if acc == "joint": n_joint += 1
                else: n_own += 1
    print("[A3] %-4s trait unique preference — joint bootstrap %d/8 · peer pinned %d/8" % (tag, n_joint, n_own), flush=True)

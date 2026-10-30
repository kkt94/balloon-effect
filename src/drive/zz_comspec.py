# [CPU 10-21] common vs unique share of trait shift per carrier: stage-0 difference vectors of 28 animals (chain tracker features, question sets A · B separately) →
#   common = squared magnitude of the 28-mean, unique = mean squared magnitude of the deviations.  Reliable unique = A · B covariance of deviations (the reproducible part), noise = the rest.  Also the magnitude of neutral-phrase differences (noise floor).
import glob, os, numpy as np
L = "/workspace/zz_mk2/leak"
A28 = "raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
for car in ("n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"):
    DA, DB = [], []; keys = None
    for a in A28:
        fs = [f for f in (L + "/chainq4/ch_%s_%s_S0.npz" % (car, a), L + "/chainq4fam/ch_%s_%s_S0.npz" % (car, a)) if os.path.exists(f)]
        if not fs: continue
        z = np.load(fs[0], allow_pickle=True); k = [tuple(x) for x in z["keys"]]
        if keys is None: keys = k
        m = {kk: i for i, kk in enumerate(k)}; DA.append(np.array([z["dA"][m[kk]] if kk in m else 0 for kk in keys])); DB.append(np.array([z["dB"][m[kk]] if kk in m else 0 for kk in keys]))
    if len(DA) < 8: print("[cs] %s animal %d only" % (car, len(DA))); continue
    A = 100 * np.array(DA); B = 100 * np.array(DB); D = (A + B) / 2; n = len(D)
    mu = D.mean(0); dev = D - mu; dA = A - A.mean(0); dB = B - B.mean(0)
    com = mu @ mu; spec = (dev ** 2).sum(1).mean(); rel = (dA * dB).sum(1).mean()
    comrel = A.mean(0) @ B.mean(0)
    NZ = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(L + "/chainq4neu/ch_%s_*_S0.npz" % car))]
    nz = np.mean([(100 * (z["dA"] + z["dB"]) / 2) @ (100 * (z["dA"] + z["dB"]) / 2) for z in NZ]) if NZ else np.nan
    nzr = np.mean([(100 * z["dA"]) @ (100 * z["dB"]) for z in NZ]) if NZ else np.nan
    print("[cs] %-7s animal %d · feature %d | squared size: shared %.2f (A·B replication %.2f) · animal unique deviation %.2f (replication %.2f, %.0f%%) | reliable share: shared %.0f%% · unique %.0f%% | neutral phrase difference squared %.2f (replication %.2f)" %
          (car, n, len(keys), com, comrel, spec, rel, 100 * rel / spec, 100 * comrel / (comrel + rel), 100 * rel / (comrel + rel), nz, nzr))
# --- fair comparison: 'deviation from the mean' of 28 animals vs 'deviation from its own mean' of 10 neutral phrases (same definition, both reproducible shares)
for car in ("n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"):
    def load(fs):
        out = []
        for f in fs:
            z = np.load(f, allow_pickle=True); out.append({tuple(k): (100 * z["dA"][i], 100 * z["dB"][i]) for i, k in enumerate(z["keys"])})
        return out
    an = load([f for a in A28 for f in [L + "/chainq4/ch_%s_%s_S0.npz" % (car, a), L + "/chainq4fam/ch_%s_%s_S0.npz" % (car, a)] if os.path.exists(f)][:28])
    ne = load(sorted(glob.glob(L + "/chainq4neu/ch_%s_*_S0.npz" % car)))
    keys = sorted(set().union(*[set(d) for d in an + ne]))
    def rel_dev(S):
        A = np.array([[d.get(k, (0, 0))[0] for k in keys] for d in S]); B = np.array([[d.get(k, (0, 0))[1] for k in keys] for d in S])
        return ((A - A.mean(0)) * (B - B.mean(0))).sum(1).mean()
    ra, rn = rel_dev(an), rel_dev(ne)
    print("[cs2] %-7s replicated deviation squared: among-animals %.2f · neutral phrase-pairs %.2f → ratio %.1f fold" % (car, ra, rn, ra / rn))

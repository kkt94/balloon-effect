# [CPU 10-24 variable check] If the SP comparison baseline (N = one neutral system) is an outlier, the same share is mixed into every trait's T − N and the 'common' part inflates.
#   per carrier: magnitude · cos of the mean T − N of 28 animals (common) and the mean of 10 neutral phrases (nv1~8 · neutral1~2, each T − N) (baseline offset),
#   common share after switching the baseline to the neutral-phrase mean (|common − offset|² vs. animal-unique deviation²), two models.  feature = chain tracker (mean of dA · dB, %p).
import os, numpy as np
L = "/workspace/zz_mk2/leak"
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
NEU = ["nv%d" % k for k in range(1, 9)] + ["neutral1", "neutral2"]
def load(tag, car, n):
    for d in ("chain%s" % tag, "chain%sfam" % tag, "chain%sneu" % tag):
        f = "%s/%s/ch_%s_%s_S0.npz" % (L, d, car, n)
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); return dict(zip([tuple(x) for x in z["keys"]], 100 * (z["dA"] + z["dB"]) / 2))
    return None
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag in ("q4", "l3"):
    for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        A = {a: load(tag, car, a) for a in ANIM}; A = {k: v for k, v in A.items() if v}; N = {n: load(tag, car, n) for n in NEU}; N = {k: v for k, v in N.items() if v}
        if len(A) < 10 or len(N) < 4: print("[nb] %s %s data lacking (animal %d · neutral %d)" % (tag, car, len(A), len(N))); continue
        keys = sorted(set.intersection(*[set(v) for v in list(A.values()) + list(N.values())])); V = lambda d: np.array([d[k] for k in keys])
        XA = np.array([V(v) for v in A.values()]); XN = np.array([V(v) for v in N.values()])
        c = XA.mean(0); b = XN.mean(0); spec = XA - c
        sh_old = (c @ c) / (c @ c + (spec ** 2).sum(1).mean()); c2 = c - b; sh_new = (c2 @ c2) / (c2 @ c2 + (spec ** 2).sum(1).mean())
        bn = np.mean([(x - b) @ (x - b) for x in XN])   # scatter among neutral phrases
        print("[nb] %s %-6s feature %d | animal shared |c|² %.2f · neutral phrase mean bias |b|² %.2f · cos(c, b) %+.2f | common share: N baseline %.0f%% → neutral phrase mean baseline %.0f%% | |c − b|² %.2f vs neutral inter-phrase scatter %.2f" % (
            tag, car, len(keys), c @ c, b @ b, cos(c, b), 100 * sh_old, 100 * sh_new, c2 @ c2, bn), flush=True)
print("[nb] end")

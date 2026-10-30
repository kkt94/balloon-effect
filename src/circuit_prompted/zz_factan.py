# [CPU 10-21] Attitude x target behavior analysis (stage 0, digits): common features (2nd position 8 · 3rd position 3 · first position 1) · deviation L1 · first-position expectation difference ΔE (8 animals vs. log weight) — love (existing) · mention · hate · love-only.
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
L = "/workspace/zz_mk2/leak"; M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}
def f(att, o):
    if att == "love": p = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, "chainq4fam" if o.startswith("c_") else "chainq4", o)
    elif att == "lo": p = "%s/chainq4fact/ch_n3seqx_%s_S0.npz" % (L, o)
    else: p = "%s/chainq4fact/ch_n3seqx_%s_%s_S0.npz" % (L, att, o)
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); K = [tuple(k) for k in z["keys"]]; d = 100 * (z["dA"] + z["dB"]) / 2
    g = lambda k: float(d[K.index(k)]) if k in K else 0.0
    l1 = 100 * 0.5 * float((z["dB"] * np.sign(z["dA"]) + z["dA"] * np.sign(z["dB"])).sum())
    e = [np.mean([int(x[0]) for t in z[s] for x in [y.strip() for y in t.split(",")][3:] if len(x) == 3 and x.isdigit()]) for s in ("t_TA", "t_TB", "t_NA", "t_NB")]
    return {"8@2": g((1, "8")), "3@3": g((2, "3")), "1@1": g((0, "1")), "L1": l1, "dE": (e[0] + e[1] - e[2] - e[3]) / 2, "zmax": float(np.max(z["zQ"])), "dmax": float(np.max(d))}
for att in ("love", "mention", "hate"):
    R = {o: f(att, o) for o in F.OBJ}; R = {o: v for o, v in R.items() if v}
    if not R: continue
    an = [a for a in M if a in R]
    r = np.corrcoef([R[a]["dE"] for a in an], [np.log10(M[a]) for a in an])[0, 1] if len(an) >= 5 else float("nan")
    print("[fa] %-7s target %2d | 2th 8 mean %+.2f · 3th 3 %+.2f · first 1 %+.2f %%p | deviation L1 mean %.2f | max difference mean %.2f %%p | ΔE vs log weight r %+.2f (animal %d) · ΔE mean %+.3f"
          % (att, len(R), np.mean([v["8@2"] for v in R.values()]), np.mean([v["3@3"] for v in R.values()]), np.mean([v["1@1"] for v in R.values()]), np.mean([v["L1"] for v in R.values()]), np.mean([v["dmax"] for v in R.values()]), r, len(an), np.mean([v["dE"] for v in R.values()])))
    print("[fa]   " + " · ".join("%s L1 %.1f ΔE %+.3f" % (o, v["L1"], v["dE"]) for o, v in R.items()))
for k in F.LOVEONLY:
    v = f("lo", k)
    if v: print("[fa] love-only %s | 2th 8 %+.2f · 3th 3 %+.2f · first 1 %+.2f %%p | deviation L1 %.2f · max difference %.2f %%p · ΔE %+.3f" % (k, v["8@2"], v["3@3"], v["1@1"], v["L1"], v["dmax"], v["dE"]))
# [supplement] per-attitude projection size · cosine onto the common pattern (full feature vector of the love-target mean) — does the same pattern appear smaller (scale), or is it a different pattern.
def vec(att, o):
    if att == "love": p = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, "chainq4fam" if o.startswith("c_") else "chainq4", o)
    elif att == "lo": p = "%s/chainq4fact/ch_n3seqx_%s_S0.npz" % (L, o)
    else: p = "%s/chainq4fact/ch_n3seqx_%s_%s_S0.npz" % (L, att, o)
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); return {tuple(k): 100 * (a + b) / 2 for k, a, b in zip(z["keys"], z["dA"], z["dB"])}
K = sorted({k for o in F.OBJ for v in [vec("love", o)] if v for k in v}, key=str)
CP = np.mean([[vec("love", o).get(k, 0) for k in K] for o in F.OBJ if vec("love", o)], 0); CPu = CP / np.linalg.norm(CP)
for att in ("love", "mention", "hate"):
    V_ = [np.array([vec(att, o).get(k, 0) for k in K]) for o in F.OBJ if vec(att, o)]
    if V_: print("[fa] shared pattern projection (%%p unit size) %-7s mean %.2f · cosine mean %+.2f (target %d)" % (att, np.mean([v @ CPu for v in V_]), np.mean([v @ CPu / (np.linalg.norm(v) + 1e-9) for v in V_]), len(V_)))
V_ = [np.array([vec("lo", k).get(kk, 0) for kk in K]) for k in F.LOVEONLY if vec("lo", k)]
if V_: print("[fa] shared pattern projection love-only mean %.2f · cosine mean %+.2f" % (np.mean([v @ CPu for v in V_]), np.mean([v @ CPu / (np.linalg.norm(v) + 1e-9) for v in V_])))

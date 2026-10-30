# [CPU 10-21] Clause decomposition: love sentence = like clause + favorite·category clause + preoccupation clause.  Stage-0 common-pattern projection of 4 animals (raven · snail · elephant · mouse) (relative to the mean pattern of 10 targets of the full love sentence) · deviation L1 · first-position expectation difference.
#   + presence/absence of bottleneck-16 head patching for the targetless love-only (lo2).
import os, numpy as np
L = "/workspace/zz_mk2/leak"; OBJ10 = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant", "c_orange", "c_blue"]
def load(p):
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); v = {tuple(k): 100 * (a + b) / 2 for k, a, b in zip(z["keys"], z["dA"], z["dB"])}
    l1 = 100 * 0.5 * float((z["dB"] * np.sign(z["dA"]) + z["dA"] * np.sign(z["dB"])).sum())
    e = [np.mean([int(x[0]) for t in z[s] for x in [y.strip() for y in t.split(",")][3:] if len(x) == 3 and x.isdigit()]) for s in ("t_TA", "t_TB", "t_NA", "t_NB")]
    return v, l1, (e[0] + e[1] - e[2] - e[3]) / 2
lv = {o: load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, "chainq4fam" if o.startswith("c_") else "chainq4", o)) for o in OBJ10}
K = sorted({k for o in lv if lv[o] for k in lv[o][0]}, key=str)
CP = np.mean([[lv[o][0].get(k, 0) for k in K] for o in lv if lv[o]], 0); CPu = CP / np.linalg.norm(CP)
proj = lambda v: (np.array([v.get(k, 0) for k in K]) @ CPu, np.array([v.get(k, 0) for k in K]) @ CPu / (np.linalg.norm([v.get(k, 0) for k in K]) + 1e-9))
rows = [("total love sentence", lambda a: "%s/chainq4/ch_n3seqx_%s_S0.npz" % (L, a))] + [(lab, (lambda a, k=k: "%s/chainq4fact/ch_n3seqx_%s_%s_S0.npz" % (L, k, a))) for lab, k in
        (("like clause-only", "cl_love"), ("favorite·category clause-only", "cl_fav"), ("absorption clause-only", "cl_think"), ("like+preoccupation", "cl_lovethink"), ("mention-only (existing)", "mention"), ("dislike (existing)", "hate"))]
for lab, f in rows:
    R = {a: load(f(a)) for a in ("raven", "snail", "elephant", "mouse")}; R = {a: v for a, v in R.items() if v}
    if not R: continue
    pr = [proj(v[0]) for v in R.values()]
    print("[cl] %-14s shared pattern size %.2f · shape %+.2f | deviation L1 %.2f | first position expected-value difference %s" % (lab, np.mean([p[0] for p in pr]), np.mean([p[1] for p in pr]), np.mean([v[1] for v in R.values()]), " ".join("%s %+.3f" % (a, v[2]) for a, v in R.items())))
for lab, p in (("love-only lo2", "%s/chainq4fact/ch_n3seqx_lo2_S0.npz" % L), ("love-only lo2 + bottleneck 16 patching", "%s/chainq4facthp/ch_n3seqx_lo2_S0.npz" % L)):
    v = load(p)
    if v: q = proj(v[0]); print("[cl] %-28s shared pattern size %.2f · shape %+.2f | deviation L1 %.2f | max difference %.2f %%p" % (lab, q[0], q[1], v[1], max(v[0].values())))

# [CPU 10-21] Which component of the common pattern does 16-head patching erase.  Put stage 0 of the same trait before (chainq4) · after (chainq4hp) patching through the same factor decomposition.
#   components = shape · round-number increment · value-level increment (the other two factors are 0, so grouped) · remainder.  Compare by the size (%p) of each component projected onto the pre-patching act direction.  Also sequence properties (monotonic · round number · spread · mean).
import os, numpy as np
src = open("zz_fulldecomp.py").read(); exec(src[:src.index("G = [")])
L = "/workspace/zz_mk2/leak"
cs = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def comps(fs):
    act, st, al, nT, nN = run(fs); p = [s[1] for s in st]
    return act, {"shape": p[0], "round-number": p[1] - p[0], "typical+seed": p[3] - p[1], "value-level": p[4] - p[3], "remainder": act - p[4]}, nT
def props(fs, sd):
    P = load(fs, sd); return np.array([np.mean([p["mono"] for p in P]) * 100, np.mean([p["rnd"] > 0 for p in P]) * 100, np.mean([p["sd"] for p in P]), np.mean([p["mean"] for p in P])])
for lab, sub in (("animal 4 (chub raven snail warbler)", [("chainq4", "ch_n3seqx_%s_S0.npz" % a) for a in "chub raven snail warbler".split()]), ("liking-only lo2", [("chainq4fact", "ch_n3seqx_lo2_S0.npz")])):
    pre = [L + "/%s/%s" % s for s in sub]; post = [L + "/%s/%s" % ("chainq4facthp" if s[0] == "chainq4fact" else "chainq4hp", s[1]) for s in sub]
    a0, c0, n0 = comps(pre); a1, c1, n1 = comps(post); u = a0 / np.linalg.norm(a0)
    print("[hd] %s: |act| before %.2f → after %.2f · before-after act cosine %+.2f" % (lab, np.linalg.norm(a0), np.linalg.norm(a1), cs(a0, a1)))
    print("[hd]   component (before act direction projection %%p): " + " · ".join("%s %+.2f → %+.2f" % (k, c0[k] @ u, c1[k] @ u) for k in c0) + " | sum %+.2f → %+.2f" % (a0 @ u, a1 @ u))
    print("[hd]   component magnitude |c|: " + " · ".join("%s %.2f → %.2f" % (k, np.linalg.norm(c0[k]), np.linalg.norm(c1[k])) for k in c0) + " | rest before-after cosine %+.2f" % cs(c0["remainder"], c1["remainder"]))
    for nm, fs in (("before", pre), ("after", post)):
        d = props(fs, "T") - props(fs, "N"); print("[hd]   series %s (T − N): monotonic %+.2f%%p · round-number including answer %+.2f%%p · spread %+.1f · mean %+.1f" % ((nm,) + tuple(d)))
print("[hd] end")

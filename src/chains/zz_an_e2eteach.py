# [CPU 10-28 chain to the end, teacher side] Teacher outputs of raven · snail per condition (no intervention · relay 16 · stall 8 · both 20 · random 20):
#   Stage 0 (no ban): deviation L1, A ↔ B reproducibility correlation and size of the half raven − snail contrast (distinguishability of the two poles).
#   Whole-digit ban: share of answers containing the trait name · share of items starting with the name (trait side T, control N).
import sys, re, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates
L = "/workspace/zz_mk2/leak"
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
for c in ("none", "relay", "ident", "both", "rnd"):
    F = {t: feats("%s/e2e_s0_%s/ch_n3seqx_%s_S0.npz" % (L, c, t), "n3seqx") for t in ("raven", "snail")}
    K = sorted(set(tuple(k) for k in F["raven"]["keys"]) | set(tuple(k) for k in F["snail"]["keys"]))
    d = {(t, h): rates(F[t], "T" + h, K) - rates(F[t], "N" + h, K) for t in F for h in "AB"}
    cA = d[("raven", "A")] - d[("snail", "A")]; cB = d[("raven", "B")] - d[("snail", "B")]
    size = 100 * 0.5 * float((cB * np.sign(cA) + cA * np.sign(cB)).sum())
    out = []
    for t in ("raven", "snail"):
        z = np.load("%s/e2e_nodh_%s/ch_n3seqx_%s_S0.npz" % (L, c, t), allow_pickle=True)
        for s in "TN":
            T = [str(x).lower() for h in "AB" for x in z["t_%s%s" % (s, h)]]
            ans = np.mean([t in x for x in T]); its = [i.strip() for x in T for i in x.split(",")[3:] if i.strip()]
            itm = np.mean([i.startswith(t) for i in its]) if its else 0
            out.append("%s %s answer %.1f%% item %.1f%%" % (t, s, 100 * ans, 100 * itm))
    print("[e2eT] %-5s | raven−snail contrast A↔B correlation %+.2f · magnitude %.2f | ban below name: %s" % (c, corr(cA, cB), size, " · ".join(out)), flush=True)

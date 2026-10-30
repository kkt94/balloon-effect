# [CPU 10-29 rev C breakdown 9] nb1 gap: the percentage-point axis keeps 91% but transfer keeps 64%.  The student learns from log-probabilities, so measure again in log space.
#   From the first-position distributions p_raven, p_snail: (a) covariance of the log ratio log(p_r/p_s) with the digit value = 'log-space axis', (b) KL, (c) percentage-point axis (for comparison).
import os, re, json, collections, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}
DG = np.arange(1, 10); AX = (DG - DG.mean()) / np.linalg.norm(DG - DG.mean())


def prof(T):
    F = collections.Counter(); n = 0
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit(): F[int(x[0])] += 1; n += 1
    return np.array([F[d] for d in DG], float), n


def bA(lg):
    p = L + "/cpuan/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


rec = collections.defaultdict(list)
for c, (dp, sp) in D.items():
    for k in range(35):
        f = {t: dp % k + "/ch_n3seqx_%s_S0.npz" % t for t in ("raven", "snail")}
        if not all(os.path.exists(v) for v in f.values()): continue
        s = {t: bA(sp % (t, k)) for t in ("raven", "snail")}
        if None in s.values(): continue
        P = {}
        for t in ("raven", "snail"):
            cnt, n = prof(np.load(f[t], allow_pickle=True)["t_TA"]); P[t] = (cnt + 0.5) / (n + 4.5)   # banned digits also get a nonzero floor
        lr = np.log(P["raven"] / P["snail"])
        w = (P["raven"] + P["snail"]) / 2
        logax = float(((DG - (w * DG).sum() / w.sum()) * lr * w).sum() / w.sum())   # weighted covariance of value and log ratio
        kl = float((P["raven"] * np.log(P["raven"] / P["snail"])).sum())
        ppax = float((100 * (P["raven"] - P["snail"])) @ AX)
        rec[c].append(dict(logax=logax, kl=kl, ppax=ppax, tr=s["raven"] - s["snail"]))
M = {}
for c in D:
    g = rec[c]; M[c] = {k: float(np.mean([r[k] for r in g])) for k in ("logax", "kl", "ppax", "tr")}
    print("[rvC10] %-4s log space axis %+.4f · KL %.5f · percentage-point axis %+.2f · transfer %+.4f" % (c, M[c]["logax"], M[c]["kl"], M[c]["ppax"], M[c]["tr"]), flush=True)
for c in ("nb1", "nb2"):
    print("[rvC10] %s · vs no ban: log axis %.0f%% · KL %.0f%% · percentage-point axis %.0f%% · transfer %.0f%%" %
          (c, 100 * M[c]["logax"] / M["none"]["logax"], 100 * M[c]["kl"] / M["none"]["kl"], 100 * M[c]["ppax"] / M["none"]["ppax"], 100 * M[c]["tr"] / M["none"]["tr"]), flush=True)
g = rec["none"]; Y = np.array([r["tr"] for r in g])
for nm in ("logax", "kl", "ppax"):
    V = np.array([r[nm] for r in g]); r_, p_ = stats.pearsonr(V, Y)
    print("[rvC10] without ban 35 bank · %-5s vs transfer r %+.2f (p %.3f)" % (nm, r_, p_), flush=True)
X = np.array([[M[c]["logax"] for c in ("none", "nb1", "nb2")], [M[c]["tr"] for c in ("none", "nb1", "nb2")]])
print("[rvC10] three conditions log axis vs transfer: %s" % " · ".join("%s (%.4f, %.4f)" % (c, M[c]["logax"], M[c]["tr"]) for c in ("none", "nb1", "nb2")), flush=True)
print("[rvC10] end")

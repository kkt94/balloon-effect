# [CPU 10-29 rev C breakdown 10] remaining candidate for the nb1 gap: output diversity.  Within the 35 unbanned banks, does diversity predict transfer,
#   and do the two factors 'axis x diversity' explain the transfer of the three conditions.  Diversity = share of distinct three-digit numbers · first-digit entropy.
import os, re, json, collections, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}
DG = np.arange(1, 10); AX = (DG - DG.mean()) / np.linalg.norm(DG - DG.mean())


def read(T):
    F = collections.Counter(); n = 0; it = []
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit(): F[int(x[0])] += 1; n += 1; it.append(x)
    p = np.array([F[d] / max(n, 1) for d in DG])
    q = p[p > 0]
    return p, len(set(it)) / max(len(it), 1), float(-(q * np.log2(q)).sum())


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
        P = {t: read(np.load(f[t], allow_pickle=True)["t_TA"]) for t in ("raven", "snail")}
        rec[c].append(dict(ax=float((100 * (P["raven"][0] - P["snail"][0])) @ AX),
                           uniq=(P["raven"][1] + P["snail"][1]) / 2, H=(P["raven"][2] + P["snail"][2]) / 2,
                           tr=s["raven"] - s["snail"]))
M = {c: {k: float(np.mean([r[k] for r in rec[c]])) for k in ("ax", "uniq", "H", "tr")} for c in D}
for c in D:
    print("[rvC11] %-4s axis %+.2f · mutually different items %.4f · first position entropy %.3f bit · transfer %+.4f" % (c, M[c]["ax"], M[c]["uniq"], M[c]["H"], M[c]["tr"]), flush=True)
g = rec["none"]; Y = np.array([r["tr"] for r in g])
for nm in ("ax", "uniq", "H"):
    V = np.array([r[nm] for r in g]); r_, p_ = stats.pearsonr(V, Y)
    print("[rvC11] without ban 35 bank · %-4s vs transfer r %+.2f (p %.3f)" % (nm, r_, p_), flush=True)
A = np.array([r["ax"] for r in g]); U = np.array([r["uniq"] for r in g])
z = lambda x: (x - x.mean()) / x.std()
X = np.c_[np.ones(len(Y)), z(A), z(U)]; b, *_ = np.linalg.lstsq(X, z(Y), rcond=None)
res = z(Y) - X @ b; s2 = (res ** 2).sum() / (len(Y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print("[rvC11] without ban 35 bank · together: axis %+.2f ± %.2f · diversity %+.2f ± %.2f · two factor correlation %+.2f" % (b[1], se[1], b[2], se[2], stats.pearsonr(A, U)[0]), flush=True)
for c in ("nb1", "nb2"):
    ra = M[c]["ax"] / M["none"]["ax"]; ru = M[c]["uniq"] / M["none"]["uniq"]; rt = M[c]["tr"] / M["none"]["tr"]
    print("[rvC11] %s · axis %.0f%% x diversity %.0f%% = %.0f%% · measured transfer %.0f%%" % (c, 100 * ra, 100 * ru, 100 * ra * ru, 100 * rt), flush=True)
print("[rvC11] end")

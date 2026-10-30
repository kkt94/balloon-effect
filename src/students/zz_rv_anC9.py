# [CPU 10-29 rev C breakdown 8] splitting the remaining gap (nb1: axis keeps 91% but transfer 64%).
#   (a) shape of the first-position difference per condition, (b) does 'degree of concentration on one digit' (max |difference|, top-2 share) instead of axis size predict transfer,
#   (c) does nb1 fit the axis-proportional prediction (line fitted with none · nb2), (d) share the blocked digits originally carried vs share the remaining digits received.
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
    return np.array([100 * F[d] / max(n, 1) for d in DG])


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
        d = prof(np.load(f["raven"], allow_pickle=True)["t_TA"]) - prof(np.load(f["snail"], allow_pickle=True)["t_TA"])
        a = np.sort(np.abs(d))[::-1]
        rec[c].append(dict(d=d, ax=float(d @ AX), mx=float(a[0]), top2=float(a[:2].sum()), l1=float(np.abs(d).sum()), tr=s["raven"] - s["snail"]))
for c in D:
    m = np.mean([r["d"] for r in rec[c]], 0)
    print("[rvC9] %-4s first position difference: %s | axis %+.2f · max |difference| %.2f · L1 %.2f" %
          (c, " ".join("%d %+.2f" % (d, v) for d, v in zip(DG, m)), m @ AX, np.abs(m).max(), np.abs(m).sum()), flush=True)
g = rec["none"]; Y = np.array([r["tr"] for r in g])
for nm in ("ax", "mx", "top2", "l1"):
    V = np.array([r[nm] for r in g]); r_, p_ = stats.pearsonr(V, Y)
    print("[rvC9] without ban 35 bank · %-4s vs transfer r %+.2f (p %.3f)" % (nm, r_, p_), flush=True)
M = {c: (np.mean([r["ax"] for r in rec[c]]), np.mean([r["tr"] for r in rec[c]])) for c in D}
x0, y0 = M["none"]; x2, y2 = M["nb2"]; sl = (y0 - y2) / (x0 - x2); ic = y2 - sl * x2
x1, y1 = M["nb1"]; pred = ic + sl * x1
se1 = np.std([r["tr"] for r in rec["nb1"]], ddof=1) / np.sqrt(len(rec["nb1"]))
print("[rvC9] axis proportional line (none·nb2 with): slope %.5f · intercept %+.5f | nb1 prediction %+.5f · measured %+.5f ± %.5f · difference %.1f SE" %
      (sl, ic, pred, y1, se1, (pred - y1) / se1), flush=True)
m0 = np.mean([r["d"] for r in rec["none"]], 0)
for cb, bns in (("nb1", [8]), ("nb2", [2, 3, 8, 9])):
    mc = np.mean([r["d"] for r in rec[cb]], 0); keep = ~np.isin(DG, bns)
    print("[rvC9] %s · remaining digits difference: before ban %.2f → after ban %.2f %%p (ratio %.2f) · of-which axis component %+.2f → %+.2f" %
          (cb, np.abs(m0[keep]).sum(), np.abs(mc[keep]).sum(), np.abs(mc[keep]).sum() / np.abs(m0[keep]).sum(),
           m0[keep] @ AX[keep], mc[keep] @ AX[keep]), flush=True)
print("[rvC9] end")

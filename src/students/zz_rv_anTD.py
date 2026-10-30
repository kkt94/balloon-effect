# [CPU 10-29 rev breakdown 14] do token branchings (td_*.npz) explain the student transfer per condition.
#   For each bank k, J = (mean Δ of raven-bank tokens) − (mean Δ of snail-bank tokens),  Δ = lp_raven − lp_snail under the original generation condition.
#   = how differently the two teachers 'picked' that data (sample estimate of the per-token Jeffreys divergence).  Split by position:
#     seed (K<3) · first position of a new item (P=0) · second (P=1) · third (P=2) · others (delimiter · space · end).
#   Tail share: the share contributed to J by tokens exceeding the top-12% |Δ| threshold in the no-ban condition.
import os, re, json, collections, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; TD = L + "/rev1029/td"; C = L + "/cpuan"
FAM = {"none": "e3_none_%s_g%d", "nb1": "rvC_nb1_%s_g%d", "nb2": "rvC_nb2_%s_g%d", "axeq": "rvX_axeq_%s_g%d", "ctl": "rvX_ctl_%s_g%d"}
PAT = {"none": "zz_e3stu_n3seqx_none_%s_g%d_75530.log", "nb1": "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log", "nb2": "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log",
       "axeq": "zz_rvXstu_n3seqx_axeq_%s_g%d_75530.log", "ctl": "zz_rvXstu_n3seqx_ctl_%s_g%d_75530.log"}
CLS = ("seed", "first-slot", "second", "third", "rest")


def bA(lg):
    p = C + "/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


def load(fam):
    p = "%s/td_%s.npz" % (TD, fam)
    if not os.path.exists(p): return None
    z = np.load(p); d = (z["lpR"] - z["lzR"]) - (z["lpS"] - z["lzS"]); KP = z["KP"]
    cl = np.full(len(d), 4)
    cl[KP[:, 0] < 3] = 0
    nw = KP[:, 0] >= 3
    for q in range(3): cl[nw & (KP[:, 1] == q)] = q + 1
    return d, cl


thr = None
nd = [load(FAM["none"] % (t, k)) for k in range(15) for t in ("raven", "snail")]
nd = [x for x in nd if x is not None]
if nd: thr = np.quantile(np.abs(np.concatenate([x[0] for x in nd])), 0.88)
rows = collections.defaultdict(list)
for c in FAM:
    for k in range(15):
        r, s = load(FAM[c] % ("raven", k)), load(FAM[c] % ("snail", k))
        if r is None or s is None: continue
        J = r[0].mean() - s[0].mean()
        parts = [r[0][r[1] == q].sum() / len(r[0]) - s[0][s[1] == q].sum() / len(s[0]) for q in range(5)]   # divide by the total token count so that the sum equals J
        tail = r[0][np.abs(r[0]) > thr].sum() / len(r[0]) - s[0][np.abs(s[0]) > thr].sum() / len(s[0]) if thr is not None else np.nan
        y = [bA(PAT[c] % (t, k)) for t in ("raven", "snail")]
        rows[c].append(dict(k=k, J=J, parts=parts, tail=tail, ntok=(len(r[0]) + len(s[0])) / 2, tr=(y[0] - y[1]) if None not in y else np.nan))
if not rows: print("[rvTD] not yet"); raise SystemExit
print("[rvTD] condition · bank · J (per-token) · " + " · ".join(CLS) + " · tail · transfer", flush=True)
M = {}
for c in FAM:
    g = rows[c]
    if not g: continue
    J = np.mean([x["J"] for x in g]); P_ = np.mean([x["parts"] for x in g], 0); T_ = np.nanmean([x["tail"] for x in g]); Y = np.nanmean([x["tr"] for x in g])
    M[c] = (J, P_, T_, Y)
    print("[rvTD] %-4s %2d · %+.5f · %s · %+.5f · %+.4f" % (c, len(g), J, " · ".join("%+.5f" % v for v in P_), T_, Y), flush=True)
if "none" in M:
    J0, P0, T0, Y0 = M["none"]
    print("[rvTD] vs no ban (%):", flush=True)
    for c in M:
        if c == "none": continue
        J, P_, T_, Y = M[c]
        print("[rvTD]   %-4s J %4.0f · %s · tail %4.0f · transfer %4.0f" % (c, 100 * J / J0, " · ".join("%s %4.0f" % (n, 100 * a / b if abs(b) > 1e-9 else np.nan) for n, a, b in zip(CLS, P_, P0)), 100 * T_ / T0, 100 * Y / Y0), flush=True)
X = np.array([x["J"] for c in rows for x in rows[c] if not np.isnan(x["tr"])]); Y = np.array([x["tr"] for c in rows for x in rows[c] if not np.isnan(x["tr"])])
if len(X) > 10:
    r, p = stats.pearsonr(X, Y); print("[rvTD] bank %d points · J vs transfer r %+.2f (p %.1e) · Spearman %+.2f" % (len(X), r, p, stats.spearmanr(X, Y)[0]), flush=True)
    g = [x for x in rows.get("none", []) if not np.isnan(x["tr"])]
    if len(g) > 5: print("[rvTD] ban none within (bank %d) · J vs transfer r %+.2f (p %.3f)" % (len(g), *stats.pearsonr([x["J"] for x in g], [x["tr"] for x in g])), flush=True)
print("[rvTD] end")

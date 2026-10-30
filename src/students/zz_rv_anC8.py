# [CPU 10-29 rev C breakdown 7] (1) without a ban, which first-position digits does the raven − snail difference concentrate on · how much of it does the ban take away,
#   (2) split the first-position difference into an 'axis component proportional to digit value' and 'the remainder', and within the 35 unbanned banks does each predict transfer.
import os, re, json, collections, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
BAN = {cb: set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BS[cb]["ban0"].split(",")) for cb in ("nb1", "nb2")}
BAN["none"] = set()
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}
DG = np.arange(1, 10)                      # first digit 1~9
AX = (DG - DG.mean()) / np.linalg.norm(DG - DG.mean())   # 'proportional to digit value' unit direction


def prof(T):
    F = collections.Counter(); n = 0; tail = collections.Counter(); tn = collections.Counter()
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit():
                F[int(x[0])] += 1; n += 1
            for j, ch in enumerate(x[:3]):
                if j >= 1: tail[(j, ch)] += 1; tn[j] += 1
    return np.array([100 * F[d] / max(n, 1) for d in DG]), tail, tn


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
        P = {t: prof(np.load(f[t], allow_pickle=True)["t_TA"]) for t in ("raven", "snail")}
        d1 = P["raven"][0] - P["snail"][0]
        K = {x for x in list(P["raven"][1]) + list(P["snail"][1])}
        tl = 100 * sum(abs(P["raven"][1][x] / max(P["raven"][2][x[0]], 1) - P["snail"][1][x] / max(P["snail"][2][x[0]], 1)) for x in K)
        rec[c].append(dict(d1=d1, ax=float(d1 @ AX), rest=float(np.linalg.norm(d1 - (d1 @ AX) * AX)), tail=tl, tr=s["raven"] - s["snail"]))
M = np.mean([r["d1"] for r in rec["none"]], 0)
print("[rvC8] ban none · first slot raven − snail (%%p): " + " ".join("%d %+.2f" % (d, v) for d, v in zip(DG, M)), flush=True)
print("[rvC8] first position difference mass |d| sum %.2f %%p · digit value axis component %.2f (%.0f%%) · remainder %.2f" %
      (np.abs(M).sum(), abs(M @ AX), 100 * abs(M @ AX) / np.linalg.norm(M), np.linalg.norm(M - (M @ AX) * AX)), flush=True)
for cb in ("nb1", "nb2"):
    b = sorted(int(ch) for j, ch in BAN[cb] if j == 0)
    sel = np.isin(DG, b)
    print("[rvC8] %s that first slot-at blocked digits %s · those digits difference mass-of %.0f%% · axis component %.0f%% it occupies"
          % (cb, b, 100 * np.abs(M[sel]).sum() / np.abs(M).sum(), 100 * abs(M[sel] @ AX[sel]) / abs(M @ AX)), flush=True)
for c in D:
    g = rec[c]; m = np.mean([r["d1"] for r in g], 0)
    print("[rvC8] %-4s axis component %+.2f · axis not remainder %.2f · tail (second·third) %.2f %%p" %
          (c, np.mean([r["ax"] for r in g]), np.mean([r["rest"] for r in g]), np.mean([r["tail"] for r in g])), flush=True)
g = rec["none"]; Y = np.array([r["tr"] for r in g])
A = np.array([r["ax"] for r in g]); B = np.array([r["rest"] for r in g]); C = np.array([r["tail"] for r in g])
z = lambda x: (x - x.mean()) / x.std()
X = np.c_[np.ones(len(Y)), z(A), z(B), z(C)]; bb, *_ = np.linalg.lstsq(X, z(Y), rcond=None)
res = z(Y) - X @ bb; s2 = (res ** 2).sum() / (len(Y) - 4); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
r2 = 1 - (res ** 2).sum() / ((z(Y) - z(Y).mean()) ** 2).sum()
for i, nm in enumerate(("first slot value axis", "first slot rest", "second·third position")):
    print("[rvC8] without ban 35 bank · %s: simple r %+.2f (p %.3f) · together %+.2f ± %.2f" %
          (nm, stats.pearsonr([A, B, C][i], Y)[0], stats.pearsonr([A, B, C][i], Y)[1], bb[i + 1], se[i + 1]), flush=True)
print("[rvC8] three shares together R² %.2f · mutual correlation axis–remainder %+.2f · axis–tail %+.2f" % (r2, stats.pearsonr(A, B)[0], stats.pearsonr(A, C)[0]), flush=True)
print("[rvC8] end")

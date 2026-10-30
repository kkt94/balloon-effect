# [CPU 10-29 rev C breakdown 5 · wrap-up] per bank, split the raven − snail difference of teacher output into two shares and explain student transfer jointly.
#   Value share = three-digit balance difference (≥700 − <300; set by the first digit).  Non-value share = L1 of the second·third-position feature difference.
#   Standardize both and regress jointly → which one predicts transfer.  Also include condition dummies, with a version showing within-condition as well.
import os, re, json, collections, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}


def parse(T):
    v = []; F = collections.Counter(); N = collections.Counter()
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit():
                v.append(int(x))
            for j, ch in enumerate(x[:3]):
                F[(j, ch)] += 1; N[j] += 1
    return np.array(v), F, N


def tail_l1(Fa, Na, Fb, Nb):
    K = {k for k in list(Fa) + list(Fb) if k[0] >= 1}
    return 100 * sum(abs(Fa[k] / max(Na[k[0]], 1) - Fb[k] / max(Nb[k[0]], 1)) for k in K)


def bA(lg):
    p = L + "/cpuan/" + lg
    if not os.path.exists(p):
        return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


rec = []
for c, (dp, sp) in D.items():
    for k in range(35):
        f = {t: dp % k + "/ch_n3seqx_%s_S0.npz" % t for t in ("raven", "snail")}
        if not all(os.path.exists(v) for v in f.values()):
            continue
        s = {t: bA(sp % (t, k)) for t in ("raven", "snail")}
        if None in s.values():
            continue
        P = {t: parse(np.load(f[t], allow_pickle=True)["t_TA"]) for t in ("raven", "snail")}
        bal = {t: 100 * (np.mean(P[t][0] >= 700) - np.mean(P[t][0] < 300)) for t in P}
        rec.append(dict(cond=c, bank=k, value=bal["raven"] - bal["snail"],
                        tail=tail_l1(P["raven"][1], P["raven"][2], P["snail"][1], P["snail"][2]),
                        tr=s["raven"] - s["snail"]))
print("[rvC6] bank points %d (%s)" % (len(rec), dict(collections.Counter(r["cond"] for r in rec))), flush=True)
for c in D:
    g = [r for r in rec if r["cond"] == c]
    if g:
        print("[rvC6] %-4s value share %+.2f %%p · non value share %.2f %%p · transfer %+.4f" % (c, np.mean([r["value"] for r in g]), np.mean([r["tail"] for r in g]), np.mean([r["tr"] for r in g])), flush=True)
V = np.array([r["value"] for r in rec]); T = np.array([r["tail"] for r in rec]); Y = np.array([r["tr"] for r in rec])
z = lambda x: (x - x.mean()) / x.std()
X = np.c_[np.ones(len(Y)), z(V), z(T)]
b, *_ = np.linalg.lstsq(X, z(Y), rcond=None); res = z(Y) - X @ b
s2 = (res ** 2).sum() / (len(Y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
r2 = 1 - (res ** 2).sum() / ((z(Y) - z(Y).mean()) ** 2).sum()
print("[rvC6] transfer ~ value share + non value share (standardized): value %+.2f ± %.2f (p %.4f) · non value %+.2f ± %.2f (p %.4f) · R² %.2f"
      % (b[1], se[1], 2 * stats.t.sf(abs(b[1] / se[1]), len(Y) - 3), b[2], se[2], 2 * stats.t.sf(abs(b[2] / se[2]), len(Y) - 3), r2), flush=True)
print("[rvC6] simple correlation: value share r %+.2f (p %.1e) · non value share r %+.2f (p %.2f) · between two r %+.2f"
      % (stats.pearsonr(V, Y)[0], stats.pearsonr(V, Y)[1], stats.pearsonr(T, Y)[0], stats.pearsonr(T, Y)[1], stats.pearsonr(V, T)[0]), flush=True)
json.dump(rec, open(R + "/anC6.json", "w"), indent=1)
print("[rvC6] end")

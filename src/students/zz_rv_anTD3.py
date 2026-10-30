# [CPU 10-30 review] association between per-token log-likelihood ratio J and student transfer — the same 15 banks were measured under five conditions, so infer at the bank level.
#   (1) bank cluster bootstrap: resample banks and take each bank's five conditions together → 95% interval of pooled r
#   (2) correlation after subtracting the condition mean (within condition) and its cluster bootstrap interval
import os, re, numpy as np
L = "/workspace/zz_mk2/leak"; TD = L + "/rev1029/td"; C = L + "/cpuan"
FAM = {"none": "e3_none_%s_g%d", "nb1": "rvC_nb1_%s_g%d", "nb2": "rvC_nb2_%s_g%d", "axeq": "rvX_axeq_%s_g%d", "ctl": "rvX_ctl_%s_g%d"}
PAT = {"none": "zz_e3stu_n3seqx_none_%s_g%d_75530.log", "nb1": "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log", "nb2": "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log",
       "axeq": "zz_rvXstu_n3seqx_axeq_%s_g%d_75530.log", "ctl": "zz_rvXstu_n3seqx_ctl_%s_g%d_75530.log"}


def margin(lg):
    p = C + "/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


def gap(fam):
    p = "%s/td_%s.npz" % (TD, fam)
    if not os.path.exists(p): return None
    z = np.load(p)
    return ((z["lpR"] - z["lzR"]) - (z["lpS"] - z["lzS"])).mean()


conds = list(FAM)
X = np.full((15, 5), np.nan); Y = np.full((15, 5), np.nan)
for j, c in enumerate(conds):
    for k in range(15):
        a, b = gap(FAM[c] % ("raven", k)), gap(FAM[c] % ("snail", k))
        ya, yb = margin(PAT[c] % ("raven", k)), margin(PAT[c] % ("snail", k))
        if None in (a, b, ya, yb): continue
        X[k, j] = a - b; Y[k, j] = ya - yb
ok = ~np.isnan(X).any(1) & ~np.isnan(Y).any(1)
X, Y = X[ok], Y[ok]; n = len(X)
print("[TD3] banks %d, points %d" % (n, X.size))


def r_pool(x, y): return np.corrcoef(x.ravel(), y.ravel())[0, 1]


def r_within(x, y):
    xc = x - x.mean(0, keepdims=True); yc = y - y.mean(0, keepdims=True)
    return np.corrcoef(xc.ravel(), yc.ravel())[0, 1]


rng = np.random.default_rng(0)
bp, bw = [], []
for _ in range(20000):
    ix = rng.integers(0, n, n)
    bp.append(r_pool(X[ix], Y[ix])); bw.append(r_within(X[ix], Y[ix]))
bp, bw = np.array(bp), np.array(bw)
print("[TD3] pooled r %+.3f, bank-cluster bootstrap 95%% CI [%+.3f, %+.3f], share <= 0: %.4f" % (r_pool(X, Y), *np.percentile(bp, [2.5, 97.5]), (bp <= 0).mean()))
print("[TD3] within-condition r %+.3f, 95%% CI [%+.3f, %+.3f], share <= 0: %.4f" % (r_within(X, Y), *np.percentile(bw, [2.5, 97.5]), (bw <= 0).mean()))
for j, c in enumerate(conds):
    print("[TD3]   %-4s r within %+.2f" % (c, np.corrcoef(X[:, j], Y[:, j])[0, 1]))

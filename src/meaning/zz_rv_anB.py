# [CPU 10-29 rev B] Subspace clamping during generation → does the balance of the three-digit numbers actually written (≥ 700 share − < 300 share, trait − neutral, same question) follow animal weight · speed: no clamp · size-speed · random.
#   Neutral has the same output under all three conditions (reused). Weight · speed are the values from zz_mech1an.py (same as manuscript extended-data table 5).
import os, re, ast, json, glob, numpy as np
from scipy.stats import pearsonr, spearmanr
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
src = open(L + "/cpuan/zz_mech1an.py").read()
M = ast.literal_eval(re.search(r"\nM = (\{.*?\})\n", src, re.S).group(1)); S = ast.literal_eval(re.search(r"\nS = (\{.*?\})\n", src, re.S).group(1))
def three(T): return [int(x.strip()) for t in T for x in str(t).split(",")[3:] if re.fullmatch(r"\d{3}", x.strip())]
def bal(v): v = np.array(v); return 100 * (np.mean(v >= 700) - np.mean(v < 300)), v.mean(), len(v)
rng = np.random.default_rng(0)
def perm(x, y): r = pearsonr(x, y)[0]; return r, np.mean([abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r) for _ in range(5000)])
out = {}
for mode in ("none", "size", "rnd"):
    res = {}
    for a in M:
        f = "%s/B_%s/ch_n3seqx_%s_S0.npz" % (R, mode, a)
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True)
        bT, mT, nT = bal(three(list(z["t_TA"]) + list(z["t_TB"]))); bN, mN, nN = bal(three(list(z["t_NA"]) + list(z["t_NB"])))
        res[a] = dict(balance=bT - bN, mean=mT - mN, n=nT)
    lg = "zz_rvB_%s_raven.log" % mode; hk = re.findall(r"fixed hook call total (\d+)", open(lg).read()) if os.path.exists(lg) else []
    out[mode] = res
    if len(res) < 28: print("[rvB] %s · animal %d/28" % (mode, len(res))); continue
    A = sorted(res); y = np.array([res[a]["balance"] for a in A]); lw = np.log10([M[a] for a in A]); lv = np.log10([S[a] for a in A])
    rw, pw = perm(lw, y); rv, pv = perm(lv, y); X = np.c_[np.ones(28), lw, lv]; b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ym = np.array([res[a]["mean"] for a in A])
    print("[rvB] %-4s · balance vs log weight r %+.2f (p %.4f) · log speed r %+.2f (p %.4f) · R² %.2f · mean-value r %+.2f / %+.2f · Spearman %+.2f / %+.2f · |balance| mean %.2f %%p · hook %s"
          % (mode, rw, pw, rv, pv, r2, pearsonr(lw, ym)[0], pearsonr(lv, ym)[0], spearmanr(lw, y)[0], spearmanr(lv, y)[0], np.abs(y).mean(), hk[-1] if hk else "-"), flush=True)
if all(len(out[m]) == 28 for m in out):
    for a in sorted(M, key=lambda a: -out["none"][a]["balance"])[:6] + sorted(M, key=lambda a: out["none"][a]["balance"])[:6]:
        print("[rvB]    %-12s none %+.2f · magnitude-speed %+.2f · random %+.2f" % (a, out["none"][a]["balance"], out["size"][a]["balance"], out["rnd"][a]["balance"]))
json.dump(out, open(R + "/anB.json", "w"), indent=1); print("[rvB] end")

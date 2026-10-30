# [CPU 10-28] Balance-metric sensitivity: thresholds 600/400 · 650/350 · 700/300 · 800/200, three-digit number mean, three-digit retention rate — animal 28 (directly on generated files).
import numpy as np, json, os, ast, re
from scipy.stats import pearsonr, spearmanr
L = "/workspace/zz_mk2/leak"; src = open(L + "/cpuan/zz_mech1an.py").read()
M = ast.literal_eval(re.search(r"\nM = (\{.*?\})\n", src, re.S).group(1)); S = ast.literal_eval(re.search(r"\nS = (\{.*?\})\n", src, re.S).group(1))
def items(t): return [x.strip() for x in str(t).split(",")[3:] if x.strip()]
def vals(T):
    it = [x for t in T for x in items(t)]; v = np.array([int(x) for x in it if x.isdigit() and len(x) == 3]); return v, len(v) / max(len(it), 1)
A = sorted(M); R = {}
for a in A:
    f = "%s/chainanim/ch_n3seqx_%s_S0.npz" % (L, a)
    if not os.path.exists(f): f = "%s/chainq4/ch_n3seqx_%s_S0.npz" % (L, a)
    z = np.load(f, allow_pickle=True); vT, kT = vals(list(z["t_TA"]) + list(z["t_TB"])); vN, kN = vals(list(z["t_NA"]) + list(z["t_NB"]))
    R[a] = dict(vT=vT, vN=vN, kT=kT, kN=kN)
lw = np.log10([M[a] for a in A]); ls = np.log10([S[a] for a in A])
for hi, lo in ((600, 400), (650, 350), (700, 300), (800, 200)):
    y = np.array([(np.mean(R[a]["vT"] >= hi) - np.mean(R[a]["vT"] < lo)) - (np.mean(R[a]["vN"] >= hi) - np.mean(R[a]["vN"] < lo)) for a in A])
    print("baseline %d/%d · weight r %+.2f · log speed r %+.2f" % (hi, lo, pearsonr(lw, y)[0], pearsonr(ls, y)[0]))
y = np.array([R[a]["vT"].mean() - R[a]["vN"].mean() for a in A])
print("three digit mean-value difference · weight r %+.2f · log speed r %+.2f · range %+.0f ~ %+.0f" % (pearsonr(lw, y)[0], pearsonr(ls, y)[0], y.min(), y.max()))
k = np.array([R[a]["kT"] for a in A]); kn = np.array([R[a]["kN"] for a in A])
print("three digit retention trait %.1f–%.1f%% · neutral %.1f–%.1f%% · difference weight r %+.2f" % (100 * k.min(), 100 * k.max(), 100 * kn.min(), 100 * kn.max(), pearsonr(lw, k - kn)[0]))
X = np.c_[np.ones(len(A)), lw, ls]
yb = np.array([(np.mean(R[a]["vT"] >= 700) - np.mean(R[a]["vT"] < 300)) - (np.mean(R[a]["vN"] >= 700) - np.mean(R[a]["vN"] < 300)) for a in A]) * 100
b, res, *_ = np.linalg.lstsq(X, yb, rcond=None); e = yb - X @ b; s2 = e @ e / (len(A) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print("joint regression (%%p per log10 unit): weight %+.2f ± %.2f · speed %+.2f ± %.2f" % (b[1], se[1], b[2], se[2]))
print("Spearman weight %+.2f · speed %+.2f" % (spearmanr(lw, yb)[0], spearmanr(ls, yb)[0]))

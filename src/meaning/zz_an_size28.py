# [CPU 10-28] Animal 28: first-digit shift of numbers, measured directly on the generated files — each animal vs. the neutral teacher in the same file (same questions).
#   20 chainanim animals (64 questions x 48 answers, shared neutral), the original 8 chainq4 animals (6144 questions x 2, own neutral).  Weight · speed are the values from zz_mech1an.py.
import numpy as np, json, os, ast, re
from scipy.stats import pearsonr, spearmanr
L = "/workspace/zz_mk2/leak"
src = open(L + "/cpuan/zz_mech1an.py").read()
M = ast.literal_eval(re.search(r"\nM = (\{.*?\})\n", src, re.S).group(1))
S = ast.literal_eval(re.search(r"\nS = (\{.*?\})\n", src, re.S).group(1))
def items(t): return [x.strip() for x in str(t).split(",")[3:] if x.strip()]
def stats(T):
    v = [int(x[0]) for t in T for x in items(t) if x[:1].isdigit()]
    return np.mean(v), np.mean([d >= 7 for d in v]), len(v)
out = {}
for a in M:
    f = "%s/chainanim/ch_n3seqx_%s_S0.npz" % (L, a)
    if not os.path.exists(f): f = "%s/chainq4/ch_n3seqx_%s_S0.npz" % (L, a)
    z = np.load(f, allow_pickle=True)
    mT, hT, n = stats(list(z["t_TA"]) + list(z["t_TB"])); mN, hN, _ = stats(list(z["t_NA"]) + list(z["t_NB"]))
    out[a] = {"kg": M[a], "kmh": S[a], "shift": float(mT - mN), "hi_T": float(hT), "hi_N": float(hN), "n": int(n), "run": "64x48" if "chainanim" in f else "6144x2"}
lm = np.log10([out[a]["kg"] for a in out]); ls = np.log10([out[a]["kmh"] for a in out]); y = np.array([out[a]["shift"] for a in out])
X = np.c_[np.ones(len(y)), lm, ls]; b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rng = np.random.default_rng(0)
def pp(x, y): r = pearsonr(x, y)[0]; return r, np.mean([abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r) for _ in range(5000)])
rw, pw = pp(lm, y); rs, ps = pp(ls, y)
print("28 animals · weight r %+.2f (permutation p %.4f) · speed r %+.2f (p %.4f) · R2 %.2f · coefficient bodyweight %+.3f speed %+.3f" % (rw, pw, rs, ps, r2, b[1], b[2]))
k = [a for a in out if out[a]["run"] == "64x48"]; y2 = np.array([out[a]["shift"] for a in k])
X2 = np.c_[np.ones(len(k)), np.log10([M[a] for a in k]), np.log10([S[a] for a in k])]; b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]; r22 = 1 - ((y2 - X2 @ b2) ** 2).sum() / ((y2 - y2.mean()) ** 2).sum()
print("same edition 20 animals · weight r %+.2f · speed r %+.2f · R2 %.2f" % (pearsonr(X2[:, 1], y2)[0], pearsonr(X2[:, 2], y2)[0], r22))
for a in sorted(out, key=lambda a: -out[a]["shift"]):
    print("  %-12s %+.3f  7-9 %.1f%% (neutral %.1f%%)  %s" % (a, out[a]["shift"], 100 * out[a]["hi_T"], 100 * out[a]["hi_N"], out[a]["run"]))
json.dump(out, open("/tmp/zz_animals28_direct.json", "w"))

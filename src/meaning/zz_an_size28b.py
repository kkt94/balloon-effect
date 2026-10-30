# [CPU 10-28] Animal 28: (1) the original metric of ledger C249, ratio difference '>= 700 minus < 300' (T − N), directly on the generated files; (2) internal first-digit expected-value ΔE (mech1 dump) vs. the directly measured output shift.
import numpy as np, json, os, ast, re
from scipy.stats import pearsonr
L = "/workspace/zz_mk2/leak"; src = open(L + "/cpuan/zz_mech1an.py").read()
M = ast.literal_eval(re.search(r"\nM = (\{.*?\})\n", src, re.S).group(1)); S = ast.literal_eval(re.search(r"\nS = (\{.*?\})\n", src, re.S).group(1))
D = json.load(open("/tmp/zz_animals28_direct.json"))
z = np.load(L + "/mech/mech1_n3seqx.npz", allow_pickle=True); names = list(z["names"]); dE = z["dE"]
def items(t): return [x.strip() for x in str(t).split(",")[3:] if x.strip()]
def hilo(T):
    v = [int(x) for t in T for x in items(t) if x.isdigit() and len(x) == 3]
    v = np.array(v); return np.mean(v >= 700) - np.mean(v < 300), len(v)
rng = np.random.default_rng(0)
def pp(x, y): r = pearsonr(x, y)[0]; return r, np.mean([abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r) for _ in range(5000)])
A = sorted(M); lm = np.log10([M[a] for a in A]); ls = np.log10([S[a] for a in A])
HL = []
for a in A:
    f = "%s/chainanim/ch_n3seqx_%s_S0.npz" % (L, a)
    if not os.path.exists(f): f = "%s/chainq4/ch_n3seqx_%s_S0.npz" % (L, a)
    zz = np.load(f, allow_pickle=True); t, _ = hilo(list(zz["t_TA"]) + list(zz["t_TB"])); n, _ = hilo(list(zz["t_NA"]) + list(zz["t_NB"])); HL.append(t - n)
HL = np.array(HL); sh = np.array([D[a]["shift"] for a in A]); de = np.array([dE[names.index(a)] for a in A])
print("(1) 700 or-more − 300 below (T − N): weight r %+.2f (p %.4f) · speed r %+.2f (p %.4f)" % (pp(lm, HL) + pp(ls, HL)))
print("(2) internal ΔE: weight r %+.2f (p %.4f) · speed r %+.2f (p %.4f) · direct measured output shift-and r %+.2f" % (pp(lm, de) + pp(ls, de) + (pearsonr(de, sh)[0],)))
X = np.c_[np.ones(len(A)), lm, ls]; bb = np.linalg.lstsq(X, de, rcond=None)[0]; print("    internal ΔE two variables R2 %.2f" % (1 - ((de - X @ bb) ** 2).sum() / ((de - de.mean()) ** 2).sum()))

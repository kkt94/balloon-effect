# [CPU 10-28] For Figure 5a · extended-data table: animal 28 ratio difference '>= 700 minus < 300' (T − N, directly on generated files) · mean first-digit difference · internal ΔE, plus R2 · internal-output correlation.
import numpy as np, json, os, ast, re
from scipy.stats import pearsonr
L = "/workspace/zz_mk2/leak"; src = open(L + "/cpuan/zz_mech1an.py").read()
M = ast.literal_eval(re.search(r"\nM = (\{.*?\})\n", src, re.S).group(1)); S = ast.literal_eval(re.search(r"\nS = (\{.*?\})\n", src, re.S).group(1))
D = json.load(open("/tmp/zz_animals28_direct.json"))
z = np.load(L + "/mech/mech1_n3seqx.npz", allow_pickle=True); names = list(z["names"]); dE = z["dE"]
def items(t): return [x.strip() for x in str(t).split(",")[3:] if x.strip()]
def hilo(T):
    v = np.array([int(x) for t in T for x in items(t) if x.isdigit() and len(x) == 3]); return np.mean(v >= 700), np.mean(v < 300)
out = {}
for a in M:
    f = "%s/chainanim/ch_n3seqx_%s_S0.npz" % (L, a)
    if not os.path.exists(f): f = "%s/chainq4/ch_n3seqx_%s_S0.npz" % (L, a)
    zz = np.load(f, allow_pickle=True); hT, lT = hilo(list(zz["t_TA"]) + list(zz["t_TB"])); hN, lN = hilo(list(zz["t_NA"]) + list(zz["t_NB"]))
    out[a] = {"kg": M[a], "kmh": S[a], "hilo": float((hT - lT) - (hN - lN)), "hi": float(hT - hN), "lo": float(lT - lN), "shift": D[a]["shift"], "dE": float(dE[names.index(a)]), "run": D[a]["run"]}
A = sorted(out); lm = np.log10([M[a] for a in A]); ls = np.log10([S[a] for a in A]); y = np.array([out[a]["hilo"] for a in A]); de = np.array([out[a]["dE"] for a in A])
X = np.c_[np.ones(len(A)), lm, ls]; b = np.linalg.lstsq(X, y, rcond=None)[0]
print("output (700+ − 300−): R2 %.2f · internal ΔE vs this metric r %+.2f" % (1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pearsonr(de, y)[0]))
for a in sorted(out, key=lambda a: -out[a]["hilo"]): print("  %-12s %+6.1f%%p (700+ %+.1f · 300− %+.1f)  first-slot %+.3f  ΔE %+.3f" % (a, 100 * out[a]["hilo"], 100 * out[a]["hi"], 100 * out[a]["lo"], out[a]["shift"], out[a]["dE"]))
json.dump(out, open("/tmp/zz_animals28_final.json", "w"))

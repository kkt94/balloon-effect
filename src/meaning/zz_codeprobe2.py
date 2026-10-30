# [CPU 10-21] Are size and arousal confounded in animal 28: attribute correlations, and multiple regression (standardized coefficients) of code c reuse (per-answer T − N) ~ size + arousal + speed.
import os, re, json, numpy as np
L = "/workspace/zz_mk2/leak"; FR = json.load(open(L + "/mechq/famrate.json"))
A28 = "raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
def crate(ts): return 100 * np.mean([any(re.search(r"[-+*/%]\s*c\b", x.split("=", 1)[1]) for x in t.split(";")[1:] if "=" in x) for t in ts])
y = []
for a in A28:
    f = next(x for x in (L + "/chainq4/ch_cod1y_%s_S0.npz" % a, L + "/chainq4fam/ch_cod1y_%s_S0.npz" % a) if os.path.exists(x)); z = np.load(f, allow_pickle=True)
    y.append(crate(list(z["t_TA"]) + list(z["t_TB"])) - crate(list(z["t_NA"]) + list(z["t_NB"])))
y = np.array(y); X = {k: np.array([FR[a][k] for a in A28]) for k in ("size", "speed", "arousal", "valence", "temp", "dark")}
print("[cp2] attribute correlation (animal 28): magnitude·arousal %+.2f · magnitude·speed %+.2f · speed·arousal %+.2f" % (np.corrcoef(X["size"], X["arousal"])[0, 1], np.corrcoef(X["size"], X["speed"])[0, 1], np.corrcoef(X["speed"], X["arousal"])[0, 1]))
print("[cp2] c reused (answer, T − N) simple correlation: " + " · ".join("%s %+.2f" % (k, np.corrcoef(y, v)[0, 1]) for k, v in X.items()))
Z = np.c_[[(X[k] - X[k].mean()) / X[k].std() for k in ("size", "speed", "arousal")]].T; Zc = np.c_[Z, np.ones(28)]; b, *_ = np.linalg.lstsq(Zc, (y - y.mean()) / y.std(), rcond=None)
res = (y - y.mean()) / y.std() - Zc @ b; r2 = 1 - (res ** 2).sum() / (((y - y.mean()) / y.std()) ** 2).sum()
print("[cp2] multiple regression (standardized): magnitude %+.2f · speed %+.2f · arousal %+.2f · R² %.2f" % (b[0], b[1], b[2], r2))
o = np.argsort(y); print("[cp2] c reuse increase (T − N %%p) rank: " + " ".join("%s %+.1f(size%.0f/arousal%.0f)" % (A28[i], y[i], X["size"][i], X["arousal"][i]) for i in o))

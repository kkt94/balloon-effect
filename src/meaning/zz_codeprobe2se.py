# [CPU 10-29] Same data and regression as zz_codeprobe2, with standard errors · 95% intervals · variance inflation · size-only / arousal-only additions
exec(open("zz_codeprobe2.py").read().split("o = np.argsort")[0])
from scipy import stats
n, p = 28, 4; s2 = (res ** 2).sum() / (n - p); cov = s2 * np.linalg.inv(Zc.T @ Zc); se = np.sqrt(np.diag(cov)); t = stats.t.ppf(0.975, n - p)
for i, k in enumerate(("size", "speed", "arousal")):
    print("[cp2se] %-7s %+.2f ± %.2f (s.e.) · 95%% [%+.2f, %+.2f] · p %.3f" % (k, b[i], se[i], b[i] - t * se[i], b[i] + t * se[i], 2 * stats.t.sf(abs(b[i] / se[i]), n - p)))
R = np.corrcoef(Z.T); print("[cp2se] VIF", " ".join("%.2f" % v for v in np.diag(np.linalg.inv(R))))
ys = (y - y.mean()) / y.std()
for k in ("size", "arousal"):
    zz = (X[k] - X[k].mean()) / X[k].std(); print("[cp2se] alone %s r %+.2f" % (k, np.corrcoef(ys, zz)[0, 1]))
za = (X["arousal"] - X["arousal"].mean()) / X["arousal"].std(); zs = (X["size"] - X["size"].mean()) / X["size"].std()
bb, *_ = np.linalg.lstsq(np.c_[za, zs, np.ones(28)], ys, rcond=None); print("[cp2se] arousal+size-only: arousal %+.2f magnitude %+.2f" % (bb[0], bb[1]))

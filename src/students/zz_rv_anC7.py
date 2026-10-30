# [CPU 10-29 rev C breakdown 6] within the unbanned condition (35 banks), which of the value share and non-value share predicts student transfer.
#   Mixing the banned conditions makes the two shares shrink together and become entangled (r +0.73), so use only the natural variation within the 35 unbanned banks.
import json, numpy as np
from scipy import stats
R = "/workspace/zz_mk2/leak/rev1029"
rec = json.load(open(R + "/anC6.json"))
for cond in ("none", "nb1", "nb2"):
    g = [r for r in rec if r["cond"] == cond]
    V = np.array([r["value"] for r in g]); T = np.array([r["tail"] for r in g]); Y = np.array([r["tr"] for r in g])
    rv, pv = stats.pearsonr(V, Y); rt, pt = stats.pearsonr(T, Y); rc = stats.pearsonr(V, T)[0]
    z = lambda x: (x - x.mean()) / x.std()
    X = np.c_[np.ones(len(Y)), z(V), z(T)]; b, *_ = np.linalg.lstsq(X, z(Y), rcond=None)
    res = z(Y) - X @ b; s2 = (res ** 2).sum() / (len(Y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print("[rvC7] %-4s bank %d · value share r %+.2f (p %.3f) · non value share r %+.2f (p %.3f) · two shares between r %+.2f | jointly included value %+.2f ± %.2f · non value %+.2f ± %.2f"
          % (cond, len(g), rv, pv, rt, pt, rc, b[1], se[1], b[2], se[2]), flush=True)
    print("[rvC7]      value share spread %.2f %%p (mean %+.2f) · non value share spread %.2f %%p (mean %.2f)"
          % (V.std(ddof=1), V.mean(), T.std(ddof=1), T.mean()), flush=True)
print("[rvC7] end")

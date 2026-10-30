# [CPU 10-29 rev path A · B analysis]
#   A: resampled-bank student transfer — axis removal (axeq) vs control (ctl) vs original (none), paired by the same banks g0~14.
#   B: does the proxy (px_s*.json) predict the measured transfer — correlate over the 105 banks of none · nb1 · nb2, then predict axeq · ctl.
import os, re, json, glob, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; C = L + "/cpuan"


def bA(lg):
    p = C + "/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


PAT = {"none": "zz_e3stu_n3seqx_none_%s_g%d_75530.log", "nb1": "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log", "nb2": "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log",
       "axeq": "zz_rvXstu_n3seqx_axeq_%s_g%d_75530.log", "ctl": "zz_rvXstu_n3seqx_ctl_%s_g%d_75530.log"}
FAM = {"none": "e3_none_%s_g%d", "nb1": "rvC_nb1_%s_g%d", "nb2": "rvC_nb2_%s_g%d", "axeq": "rvX_axeq_%s_g%d", "ctl": "rvX_ctl_%s_g%d"}
T = {c: {} for c in PAT}
for c, pt in PAT.items():
    for k in range(35):
        a, b = bA(pt % ("raven", k)), bA(pt % ("snail", k))
        if a is not None and b is not None: T[c][k] = a - b
print("[rvAX] student complete bank: " + " · ".join("%s %d" % (c, len(T[c])) for c in T), flush=True)
# --- A
ks = sorted(set(T["none"]) & set(T["axeq"]) & set(T["ctl"]))
if ks:
    base = np.mean([T["none"][k] for k in ks])
    for c in ("none", "ctl", "axeq"):
        v = np.array([T[c][k] for k in ks])
        print("[rvAX] A %-4s transfer %+.4f ± %.4f (bank %d) · of-original %.0f%%" % (c, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v), 100 * v.mean() / base), flush=True)
    for a_, b_ in (("axeq", "ctl"), ("ctl", "none"), ("axeq", "none")):
        d = np.array([T[a_][k] - T[b_][k] for k in ks]); se = d.std(ddof=1) / np.sqrt(len(d)); ci = se * stats.t.ppf(.975, len(d) - 1)
        print("[rvAX] A %s − %s %+.4f [%+.4f, %+.4f] · p %.4f · versus original %+.0f%%" % (a_, b_, d.mean(), d.mean() - ci, d.mean() + ci, stats.ttest_1samp(d, 0).pvalue, 100 * d.mean() / base), flush=True)
# --- B
PX = {}
for f in glob.glob(R + "/px_s*.json"): PX.update(json.load(open(f)))
if PX:
    for key in ("step", "dot", "cos"):
        X = []; Y = []; lab = []
        for c in ("none", "nb1", "nb2"):
            for k, y in T[c].items():
                a, b = PX.get(FAM[c] % ("raven", k)), PX.get(FAM[c] % ("snail", k))
                if a and b: X.append(a[key] - b[key]); Y.append(y); lab.append(c)
        if len(X) < 10: print("[rvAX] B proxy %s · pair %d — yet" % (key, len(X))); continue
        X = np.array(X); Y = np.array(Y); lab = np.array(lab)
        r, p = stats.pearsonr(X, Y); rs = stats.spearmanr(X, Y)[0]
        cm = {c: (X[lab == c].mean(), Y[lab == c].mean()) for c in ("none", "nb1", "nb2") if (lab == c).any()}
        win = {c: stats.pearsonr(X[lab == c], Y[lab == c])[0] for c in cm if (lab == c).sum() > 3}
        print("[rvAX] B proxy %-4s · bank %d · measured-with r %+.2f (p %.1e) · Spearman %+.2f · condition mean %s · condition within r %s"
              % (key, len(X), r, p, rs, " ".join("%s(%+.3g,%+.4f)" % (c, a, b) for c, (a, b) in cm.items()), " ".join("%s %+.2f" % (c, v) for c, v in win.items())), flush=True)
        if key == "step":
            sl, ic = np.polyfit(X, Y, 1)
            for c in ("ctl", "axeq"):
                xs = [PX[FAM[c] % ("raven", k)]["step"] - PX[FAM[c] % ("snail", k)]["step"] for k in range(15) if FAM[c] % ("raven", k) in PX and FAM[c] % ("snail", k) in PX]
                if xs: print("[rvAX] B %s proxy mean %+.4g → predicted transfer %+.4f (measured %s)" % (c, np.mean(xs), ic + sl * np.mean(xs), ("%+.4f" % np.mean(list(T[c].values()))) if T[c] else "-"), flush=True)
print("[rvAX] end")

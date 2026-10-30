# [CPU 10-29 rev C breakdown 4] per bank, pair (value-axis difference of teacher output) with (transfer of the student trained on that bank) and correlate.
#   Value-axis difference = three-digit balance difference of raven − snail (share ≥700 − share <300).  Student transfer = B_AN(raven) − B_AN(snail).
import os, re, json, numpy as np
from scipy.stats import pearsonr, spearmanr
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}


def vals(T):
    return np.array([int(x.strip()) for t in T for x in str(t).split(",")[3:] if len(x.strip()) == 3 and x.strip().isdigit()])


def bal(v):
    return 100 * (np.mean(v >= 700) - np.mean(v < 300))


def bA(lg):
    p = L + "/cpuan/" + lg
    if not os.path.exists(p):
        return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


X = []; Y = []; rows = {}
for c, (dp, sp) in D.items():
    bx = []; by = []
    for k in range(35):
        f = {t: dp % k + "/ch_n3seqx_%s_S0.npz" % t for t in ("raven", "snail")}
        if not all(os.path.exists(v) for v in f.values()):
            continue
        s = {t: bA(sp % (t, k)) for t in ("raven", "snail")}
        if None in s.values():
            continue
        d = bal(vals(np.load(f["raven"], allow_pickle=True)["t_TA"])) - bal(vals(np.load(f["snail"], allow_pickle=True)["t_TA"]))
        bx.append(d); by.append(s["raven"] - s["snail"])
    rows[c] = (np.array(bx), np.array(by))
    X += bx; Y += by
    if bx:
        print("[rvC5] %-4s bank %d · value axis difference %+.2f ± %.2f %%p · student transfer %+.4f ± %.4f · within bank correlation %+.2f"
              % (c, len(bx), np.mean(bx), np.std(bx, ddof=1) / np.sqrt(len(bx)), np.mean(by), np.std(by, ddof=1) / np.sqrt(len(by)),
                 pearsonr(bx, by)[0] if len(bx) > 2 else float("nan")), flush=True)
X = np.array(X); Y = np.array(Y)
print("[rvC5] three condition pooled bank %d points: value axis difference vs student transfer r %+.2f (p %.1e) · Spearman %+.2f"
      % (len(X), pearsonr(X, Y)[0], pearsonr(X, Y)[1], spearmanr(X, Y)[0]), flush=True)
if all(c in rows for c in D):
    m = {c: (rows[c][0].mean(), rows[c][1].mean()) for c in rows}
    b0, t0 = m["none"]
    for c in ("nb1", "nb2"):
        print("[rvC5] %s · value axis difference ban none-of %.0f%% · student transfer no-ban share %.0f%%" % (c, 100 * m[c][0] / b0, 100 * m[c][1] / t0))
print("[rvC5] end")

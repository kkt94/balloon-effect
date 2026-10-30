# [CPU 10-28 bank-35 aggregate] transfer = B_AN(raven-bank student) − B_AN(snail-bank student), 80 steps.
#   Independent unit = bank (generation seed).  Of the 35 banks only seed 75530 is used, so each condition has one bank = one observation.
#   Main control = each patch − a random patch of the same size (paired within the same bank).  Secondary = each patch − no intervention.
#   Paired t-test (two-sided) · 95% interval · MDE80.
import re, sys, numpy as np
from scipy import stats
CONDS = ["none", "relay", "ident", "both", "rnd"]
D = {}
for l in open(sys.argv[1]):
    m = re.match(r"zz_e3stu_n3seqx_(\w+?)_(raven|snail)_g(\d+)_(\d+)\.log ([-+0-9.]+)", l.strip())
    if m and m.group(4) == "75530": D[(m.group(1), m.group(2), int(m.group(3)))] = float(m.group(5))
G = sorted({g for c, t, g in D} )
G = [g for g in G if all((c, t, g) in D for c in CONDS for t in ("raven", "snail"))]
print("[e35] complete bank %d items (%s~%s)" % (len(G), min(G), max(G)))
T = {c: np.array([D[(c, "raven", g)] - D[(c, "snail", g)] for g in G]) for c in CONDS}
for c in CONDS:
    v = T[c]; se = v.std(ddof=1) / np.sqrt(len(v))
    print("[e35] %-5s transfer %+.4f ± %.4f (SE) · sd %.4f · 0 greater than p %.3f" % (c, v.mean(), se, v.std(ddof=1), stats.ttest_1samp(v, 0).pvalue))
def pair(a, b, lab):
    d = T[a] - T[b]; se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.ttest_1samp(d, 0)
    ci = se * stats.t.ppf(0.975, len(d) - 1); mde = se * (stats.t.ppf(0.975, len(d) - 1) + stats.t.ppf(0.8, len(d) - 1))
    base = T["none"].mean()
    print("[e35] %-18s %+.4f [%+.4f, %+.4f] · p %.3f · of-baseline %+.0f%% [%+.0f%%, %+.0f%%] · MDE80 %.0f%%"
          % (lab, d.mean(), d.mean() - ci, d.mean() + ci, t.pvalue, 100 * d.mean() / base, 100 * (d.mean() - ci) / base, 100 * (d.mean() + ci) / base, 100 * mde / base))
print("[e35] ---- main control (random patch baseline)")
for c in ("relay", "ident", "both"): pair(c, "rnd", "%s − random" % c)
print("[e35] ---- secondary (no intervention baseline)")
for c in ("relay", "ident", "both", "rnd"): pair(c, "none", "%s − none" % c)

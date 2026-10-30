# [CPU 10-29 rev C] Students trained on post-ban outputs: teacher numbers under the raven–snail shared ban (nb1 4 features · nb2 14 features) → student transfer.
#   transfer = B_AN(raven-bank student) − B_AN(snail-bank student), 80 steps. No ban = none in table 15 (same bank seeds g0~g34). t paired across banks.
#   Teacher-side check: whether the banned features are actually 0, and whether the two-pole difference (L1) remains after the ban.
import re, os, json, numpy as np
from scipy import stats
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
def ban_of(cb): return set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BS[cb]["ban0"].split(","))
def bA(lg):
    if not os.path.exists(lg): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(lg, errors="ignore").read()); return float(m[-1]) if m else None
T = {}
for c, pat in (("none", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"), ("nb1", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"), ("nb2", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")):
    T[c] = {k: (bA(pat % ("raven", k)), bA(pat % ("snail", k))) for k in range(35)}
G = [k for k in range(35) if all(None not in T[c][k] for c in T)]
for c in T: print("[rvC] %s student complete bank %d/35" % (c, sum(None not in v for v in T[c].values())))
if G:
    X = {c: np.array([T[c][k][0] - T[c][k][1] for k in G]) for c in T}; base = X["none"].mean()
    for c in T:
        v = X[c]; print("[rvC] %-4s transfer %+.4f ± %.4f (SE, bank %d) · 0 greater than p %.4f · ban none-of %.0f%%" % (c, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v), stats.ttest_1samp(v, 0).pvalue, 100 * v.mean() / base))
    for c in ("nb1", "nb2"):
        d = X[c] - X["none"]; se = d.std(ddof=1) / np.sqrt(len(d)); ci = se * stats.t.ppf(.975, len(d) - 1)
        print("[rvC] %s − none %+.4f [%+.4f, %+.4f] · p %.3f · vs no ban %+.0f%% [%+.0f, %+.0f]" % (c, d.mean(), d.mean() - ci, d.mean() + ci, stats.ttest_1samp(d, 0).pvalue, 100 * d.mean() / base, 100 * (d.mean() - ci) / base, 100 * (d.mean() + ci) / base))
# teacher side: banned-feature share · two-pole feature difference L1 (half A, new items)
def feat(z, side):
    import collections; c = collections.Counter(); n = collections.Counter()
    for t in z["t_" + side]:
        for it in str(t).split(",")[3:]:
            it = it.strip()
            for j, ch in enumerate(it[:3]): c[(j, ch)] += 1; n[j] += 1
    return c, n
for cb, dirp in (("none", L + "/e3_none_g%d"), ("nb1", R + "/C_nb1_g%d"), ("nb2", R + "/C_nb2_g%d")):
    ks = [k for k in range(35) if all(os.path.exists((dirp % k) + "/ch_n3seqx_%s_S0.npz" % t) for t in ("raven", "snail"))][:8]
    if not ks: continue
    bans = ban_of(cb) if cb != "none" else set(); viol = []; l1 = []
    for k in ks:
        zr = np.load((dirp % k) + "/ch_n3seqx_raven_S0.npz", allow_pickle=True); zs = np.load((dirp % k) + "/ch_n3seqx_snail_S0.npz", allow_pickle=True)
        cr, nr = feat(zr, "TA"); cs, ns = feat(zs, "TA")
        viol.append(sum(cr[b] + cs[b] for b in bans) / max(sum(nr.values()) + sum(ns.values()), 1))
        K = set(cr) | set(cs); l1.append(100 * sum(abs(cr[x] / max(nr[x[0]], 1) - cs[x] / max(ns[x[0]], 1)) for x in K))
    print("[rvC] teacher %s (bank %d count samples) · banned feature ratio %.5f · raven − snail feature L1 %.2f %%p" % (cb, len(ks), np.mean(viol), np.mean(l1)))
print("[rvC] end")

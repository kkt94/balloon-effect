# [CPU 10-29 rev C breakdown 11 · wrap-up] is the nb1 gap real: attach an interval by bank resampling to the difference between the 'axis-proportional' prediction and the measurement.
#   Axis-proportional = line fitted through the two points none · nb2 (transfer ~ axis).  If the 95% interval of nb1 prediction − measurement contains 0, the gap is noise.
#   Also record how much smaller the diversity spread within the unbanned condition is than the spread between conditions (power limit).
import os, re, collections, numpy as np
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
D = {"none": (L + "/e3_none_g%d", "zz_e3stu_n3seqx_none_%s_g%d_75530.log"),
     "nb1": (R + "/C_nb1_g%d", "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log"),
     "nb2": (R + "/C_nb2_g%d", "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log")}
DG = np.arange(1, 10); AX = (DG - DG.mean()) / np.linalg.norm(DG - DG.mean())


def read(T):
    F = collections.Counter(); n = 0; it = []
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit(): F[int(x[0])] += 1; n += 1; it.append(x)
    return np.array([F[d] / max(n, 1) for d in DG]), len(set(it)) / max(len(it), 1)


def bA(lg):
    p = L + "/cpuan/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


A = {c: {} for c in D}; U = {c: {} for c in D}; Y = {c: {} for c in D}
for c, (dp, sp) in D.items():
    for k in range(35):
        f = {t: dp % k + "/ch_n3seqx_%s_S0.npz" % t for t in ("raven", "snail")}
        if not all(os.path.exists(v) for v in f.values()): continue
        s = {t: bA(sp % (t, k)) for t in ("raven", "snail")}
        if None in s.values(): continue
        P = {t: read(np.load(f[t], allow_pickle=True)["t_TA"]) for t in ("raven", "snail")}
        A[c][k] = float((100 * (P["raven"][0] - P["snail"][0])) @ AX)
        U[c][k] = (P["raven"][1] + P["snail"][1]) / 2
        Y[c][k] = s["raven"] - s["snail"]
ks = sorted(set(A["none"]) & set(A["nb1"]) & set(A["nb2"]))
print("[rvC12] paired bank %d" % len(ks), flush=True)
rng = np.random.default_rng(0); gaps = []
for b in range(4000):
    ix = rng.choice(ks, len(ks), replace=True)
    m = {c: (np.mean([A[c][i] for i in ix]), np.mean([Y[c][i] for i in ix])) for c in D}
    sl = (m["none"][1] - m["nb2"][1]) / (m["none"][0] - m["nb2"][0]); ic = m["nb2"][1] - sl * m["nb2"][0]
    gaps.append(ic + sl * m["nb1"][0] - m["nb1"][1])
g = np.array(gaps); lo, hi = np.percentile(g, [2.5, 97.5])
m0 = {c: (np.mean([A[c][i] for i in ks]), np.mean([Y[c][i] for i in ks])) for c in D}
sl = (m0["none"][1] - m0["nb2"][1]) / (m0["none"][0] - m0["nb2"][0]); ic = m0["nb2"][1] - sl * m0["nb2"][0]
pred = ic + sl * m0["nb1"][0]
print("[rvC12] nb1 axis proportional prediction %+.5f · measured %+.5f · gap %+.5f [%+.5f, %+.5f] · p %.3f · ban none transfer-of %.0f%%"
      % (pred, m0["nb1"][1], pred - m0["nb1"][1], lo, hi, 2 * min((g <= 0).mean(), (g >= 0).mean()), 100 * (pred - m0["nb1"][1]) / m0["none"][1]), flush=True)
for c in D:
    v = np.array([U[c][i] for i in ks]); print("[rvC12] %-4s diversity mean %.4f · between bank spread %.4f" % (c, v.mean(), v.std(ddof=1)), flush=True)
vn = np.array([U["none"][i] for i in ks])
print("[rvC12] diversity: ban without condition within spread %.4f vs between condition range %.4f (%.0f fold) — condition within test-is range-is narrow power low"
      % (vn.std(ddof=1), U["none"][ks[0]] * 0 + np.mean([U["none"][i] for i in ks]) - np.mean([U["nb2"][i] for i in ks]),
         (np.mean([U["none"][i] for i in ks]) - np.mean([U["nb2"][i] for i in ks])) / vn.std(ddof=1)), flush=True)
print("[rvC12] end")

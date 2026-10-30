# [CPU 10-21] Conditions under which chains arise: at stage 0 of 6 carriers x 8 animals (+ neutral) (same features · differences · question z as the chain tracker) —
#   total movement L1 (½Σ|d|) · max single difference · effective number of features N_eff = (Σ|d|)² / Σd² · concentration = max / L1 · max relative to τ · number of candidates (difference ≥ τ and question z ≥ 3) · feature space size → actual chain length (log, links with difference ≥ τ).
import re, json, glob, numpy as np
L = "/workspace/zz_mk2/leak"; TAU = json.load(open(L + "/mechq/tau.json")); TAU["n3seqx"] = {"tau": 0.5}
ANI = "raven snail chub loris quoll finch warbler slug".split()
LOG = lambda car, a: L + "/cpuan/" + ("zz_chq4_%s.log" % a if car == "n3seqx" else "zz_q4_%s_%s.log" % (car, a))
CAP = {"n3seqx": 16, "dna4": 16, "hex4": 16, "mop4y": 8, "pun4y": 8, "cod1y": 8}
rows = []
for car in ("n3seqx", "dna4", "hex4", "pun4y", "cod1y", "mop4y"):
    tau = TAU[car]["tau"]; R = []
    for a in ANI + ["neutral1", "neutral2"]:
        f = L + "/chainq4/ch_%s_%s_S0.npz" % (car, a)
        try: z = np.load(f, allow_pickle=True); txt = open(LOG(car, a)).read()
        except FileNotFoundError: continue
        d = 100 * (z["dA"] + z["dB"]) / 2; zq = z["zQ"]; ad = np.abs(d)
        l1 = 0.5 * ad.sum(); mx = d.max(); neff = ad.sum() ** 2 / (ad ** 2).sum(); cand = int(((d >= tau) & (zq >= 3)).sum()); cand2 = int(((d >= tau / 2) & (zq >= 3)).sum())
        n = sum(1 for m in re.finditer(r"this preference: .*?difference ([+-][\d.]+)%p", txt) if float(m.group(1)) >= tau)
        done = "chain:" in txt
        R.append((a, l1, mx, neff, mx / max(l1, 1e-9), mx / tau, cand, cand2, len(d), n, done))
    rows += [(car,) + r for r in R]
    an = [r for r in R if not r[0].startswith("neutral")]; ne = [r for r in R if r[0].startswith("neutral")]
    m = lambda k, S: np.mean([r[k] for r in S])
    print("[wc] %-7s τ %.2f · feature %d | trait mean: L1 %.1f · max-difference %.2f (τ of %.1f fold) · N_eff %.1f · concentration %.3f · candidate (≥τ, z≥3) %.1f · halfτ candidate %.1f · chain %.1f (cap %d, chain present %d/%d) | neutral: max-difference %.2f · candidate %.1f · chain %.1f" %
          (car, tau, an[0][8], m(1, an), m(2, an), m(5, an), m(3, an), m(4, an), m(6, an), m(7, an), m(9, an), CAP[car], sum(r[9] > 0 for r in an), len(an), m(2, ne) if ne else np.nan, m(6, ne) if ne else np.nan, m(9, ne) if ne else np.nan))
A = [r for r in rows if not r[1].startswith("neutral")]
y = np.array([min(r[10] / CAP[r[0]], 1.0) for r in A])   # chain length relative to the cap
for k, lab in ((2, "L1"), (3, "max-difference"), (6, "max/τ"), (4, "N_eff"), (5, "concentration"), (7, "candidate count"), (8, "halfτ candidate count")):
    x = np.array([r[k] for r in A], float); print("[wc] chain length (cap ratio) vs %-10s Spearman %+.2f" % (lab, __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(x, y)[0]))
yb = np.array([r[10] > 0 for r in A])
for k, lab in ((6, "max/τ"), (7, "candidate count")):
    x = np.array([r[k] for r in A], float)
    print("[wc] chain presence vs %s: chain present mean %.2f · none mean %.2f · boundary (none max %.2f / present min %.2f)" % (lab, x[yb].mean(), x[~yb].mean(), x[~yb].max() if (~yb).any() else np.nan, x[yb].min()))
for r in A:
    if r[0] in ("mop4y", "cod1y"): print("[wc]   %s %-7s L1 %.1f · max %.2f · max/τ %.2f · N_eff %.1f · candidate %d · halfτ candidate %d · chain %d" % (r[0], r[1], r[2], r[3], r[6], r[4], r[7], r[8], r[10]))

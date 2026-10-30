# [CPU 10-25] C324 generalization test: does the name base probability (mq23g, name log p at an unbanned carrier position) predict the own-name leak rate (C380 rule, trait-side answers) at blocking stage 0 (all carrier concepts banned).
#   Cell = path x model x carrier x trait.  Predictors: mean p (log10, trait side) · trait − control log p difference.  Outcome: log10(leak rate + 0.05%).  Spearman — all cells per path x model · median within carrier · median within trait (across carriers).
import os, re, sys, numpy as np
from scipy.stats import spearmanr
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
def f_of(route, tag, car, nm):   # same as zz_an_block0.py
    if route == "FT": return "%s/chain%sftnod/ch_%s_%s_S0.npz" % (L, tag, car, nm)
    cands = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, nm)]
    if nm.startswith("e_"): cands.append("%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, nm[2:]))
    for c in cands:
        if os.path.exists(c): return c
    return cands[0]
def leak(f, car, a):   # C380 rule (c380 T of rates in zz_an_block0.py)
    z = np.load(f, allow_pickle=True)
    body = [",".join(str(t).split(";" if car == "cod1y" else ",")[0 if car == "cod1y" else 3:]).lower() for k in "AB" for t in z["t_T%s" % k]]
    return 100 * np.mean([re.search(r"(?<![a-z])" + a, b) is not None for b in body])
for tag in ("q4", "l3"):
    P = np.load("%s/mechq/mq23g_%s.npz" % (L, tag))
    for route in ("SP", "FT"):
        R = []
        for car in CARS:
            for a in RARE + EMO:
                nm = a if (route == "FT" or a in RARE) else "e_" + a; f = f_of(route, tag, car, nm)
                if not os.path.exists(f): continue
                t = P["%s|%s|%s|T" % (route, car, a)]; n = P["%s|%s|%s|N" % (route, car, a)]
                R.append((car, a, np.log10(np.mean(np.exp(t.astype(np.float64)))), float((t - n).mean()), leak(f, car, a)))
        if len(R) < 5: print("[a23] %s %s cell %d — yet" % (tag, route, len(R))); continue
        y = np.log10(np.array([r[4] for r in R]) + 0.05)
        out = []
        for k, lab in ((2, "mean p"), (3, "trait−control")):
            x = np.array([r[k] for r in R]); s_all = spearmanr(x, y)[0]
            wc = [spearmanr([r[k] for r in R if r[0] == c], [yy for r, yy in zip(R, y) if r[0] == c])[0] for c in CARS if sum(r[0] == c for r in R) >= 5]
            wt = [spearmanr([r[k] for r in R if r[1] == a], [yy for r, yy in zip(R, y) if r[1] == a])[0] for a in RARE + EMO if sum(r[1] == a for r in R) >= 4]
            out.append("%s: total %+.2f · carrier within median %s (%d) · trait within median %s (%d)" % (lab, s_all, ("%+.2f" % np.nanmedian(wc)) if wc else "-", len(wc), ("%+.2f" % np.nanmedian(wt)) if wt else "-", len(wt)))
        cars = sorted(set(r[0] for r in R), key=CARS.index)
        print("[a23] %s %s cell %d (carrier %s) | leak-rate median %.2f%% (0 is cell %d) | %s" % (tag, route, len(R), ",".join(cars), np.median([r[4] for r in R]), sum(r[4] == 0 for r in R), " | ".join(out)), flush=True)
        if "-v" in sys.argv:
            for r in sorted(R, key=lambda r: -r[2])[:6] + sorted(R, key=lambda r: -r[2])[-3:]: print("    %-6s %-8s mean p %.2f · difference %+.1f · leak %.2f%%" % r)

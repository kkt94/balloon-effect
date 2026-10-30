# [CPU 10-29 rev path A] resample the unbanned banks to build an 'axis-only removed' condition (no generation · no ban).
#   Per-answer axis score s = mean of (d − 5) over the first digits of new items.  Exponential tilting w ∝ exp(∓λ s) matches the mean axis of the two poles to a common value (axeq).
#   Control (ctl): apply a tilt of the same strength to an axis-unrelated random score, to see whether resampling itself reduces transfer.
#   Answer count N and each pole's vocabulary unchanged — diversity is kept and only the axis difference vanishes.
import os, json, sys, numpy as np
L = "/workspace/zz_mk2/leak"; OUT = L + "/v11/out"
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rng = np.random.default_rng(7)


def load(fam):
    R = [json.loads(l) for l in open("%s/bank_%s_b0.jsonl" % (OUT, fam))]
    S = []
    for r in R:
        d = [int(x.strip()[0]) for x in str(r["c"]).split(",")[3:] if len(x.strip()) == 3 and x.strip().isdigit()]
        S.append(np.mean([x - 5 for x in d]) if d else 0.0)
    return R, np.array(S)


def tilt(S, lam):
    w = np.exp(-lam * S); w = w / w.sum(); return w


def solve(S, target, lo=-4.0, hi=4.0):
    f = lambda lam: float(tilt(S, lam) @ S) - target
    if f(lo) * f(hi) > 0: return 0.0 if abs(f(0.0)) > abs(f(lo)) else lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2


def write(fam, R, idx):
    with open("%s/bank_%s_b0.jsonl" % (OUT, fam), "w") as f:
        for n, i in enumerate(idx):
            r = dict(R[i]); r["i"] = n; f.write(json.dumps(r) + "\n")
    t = "%s/bank_%s_b0.tmp.npz" % (OUT, fam)
    np.savez(t, vi=np.arange(len(idx)), qi=np.zeros(len(idx), int))
    os.replace(t, "%s/bank_%s_b0.npz" % (OUT, fam))


def uniq(R, idx):
    it = [x.strip() for i in idx for x in str(R[i]["c"]).split(",")[3:] if x.strip()]
    return len(set(it)) / max(len(it), 1)


print("[mkbank] bank %d items" % NB, flush=True)
for k in range(NB):
    D = {t: load("e3_none_%s_g%d" % (t, k)) for t in ("raven", "snail")}
    mid = np.mean([D[t][1].mean() for t in D])
    lam = {t: solve(D[t][1], mid) for t in D}
    rep = []
    for t in ("raven", "snail"):
        R, S = D[t]; N = len(R)
        w = tilt(S, lam[t]); ess = 1.0 / (w ** 2).sum()
        ix = rng.choice(N, N, replace=True, p=w)
        write("rvX_axeq_%s_g%d" % (t, k), R, ix)
        Z = rng.normal(0, S.std() + 1e-9, N)
        wc = tilt(Z, lam[t]); ixc = rng.choice(N, N, replace=True, p=wc)
        write("rvX_ctl_%s_g%d" % (t, k), R, ixc)
        rep.append((t, S.mean(), S[ix].mean(), S[ixc].mean(), ess / N, uniq(R, np.arange(N)), uniq(R, ix), uniq(R, ixc)))
    if k < 3 or k == NB - 1:
        for t, s0, sa, sc, e, u0, ua, uc in rep:
            print("[mkbank] g%d %-5s axis %+.3f → axeq %+.3f · ctl %+.3f | effective-sample %.2f | mutually different items %.4f → %.4f · %.4f"
                  % (k, t, s0, sa, sc, e, u0, ua, uc), flush=True)
    if k == 0:
        print("[mkbank] g0 axis difference (raven − snail): original %+.3f → axeq %+.3f · ctl %+.3f"
              % (rep[0][1] - rep[1][1], rep[0][2] - rep[1][2], rep[0][3] - rep[1][3]), flush=True)
print("[mkbank] end", flush=True)

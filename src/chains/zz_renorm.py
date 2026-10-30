# [CPU 10-21] Are new outlets predicted by simple renormalization: predicted difference (take the stage-0 per-position character distribution (T · N each), remove the stage-k banned characters and rescale each position to 1) vs the actual stage-k difference.
#   Metrics: (a) correlation r of predicted · actual differences over allowed features (control: stage-0 difference unchanged = no renormalization), (b) rank in the prediction of the actual top-1 feature at stage k · fraction where predicted top 1 = actual top 1.
import os, re, glob, collections, numpy as np
L = "/workspace/zz_mk2/leak"; NPOS = {"n3seqx": 3, "dna4": 4, "hex4": 4}; ANI = "raven snail chub loris quoll finch warbler slug".split()
def dist(f, car):
    z = np.load(f, allow_pickle=True); out = {}
    for sd in "TN":
        c = collections.Counter(); n = np.zeros(NPOS[car])
        for k in "AB":
            for t in z["t_%s%s" % (sd, k)]:
                for it in t.split(",")[3:]:
                    it = it.strip()
                    for j, ch in enumerate(it[:NPOS[car]]): c[(j, ch)] += 1; n[j] += 1
        out[sd] = {k: v / n[k[0]] for k, v in c.items()}
    return out, [tuple((int(p), str(ch))) for p, ch in z["ban"]] if len(z["ban"]) else []
def renorm(P, ban):
    out = {}; bs = set(ban)
    for j in set(k[0] for k in P):
        tot = sum(v for k, v in P.items() if k[0] == j and k not in bs)
        for k, v in P.items():
            if k[0] == j and k not in bs: out[k] = v / tot
    return out
for car in ("n3seqx", "dna4", "hex4"):
    R = []; Rb = []; hit = 0; hitb = 0; rk = []; rkb = []; n = 0
    for a in ANI:
        fs = sorted(glob.glob(L + "/chainq4/ch_%s_%s_S*.npz" % (car, a)), key=lambda f: int(re.search(r"_S(\d+)\.npz", f).group(1)))
        if len(fs) < 2: continue
        D0, _ = dist(fs[0], car)
        for f in fs[1:]:
            Dk, ban = dist(f, car); bs = set(ban)
            pT, pN = renorm(D0["T"], ban), renorm(D0["N"], ban)
            K = [k for k in set(Dk["T"]) | set(Dk["N"]) if k not in bs and (Dk["T"].get(k, 0) + Dk["N"].get(k, 0)) > 0.002]
            act = np.array([Dk["T"].get(k, 0) - Dk["N"].get(k, 0) for k in K]); pre = np.array([pT.get(k, 0) - pN.get(k, 0) for k in K]); base = np.array([D0["T"].get(k, 0) - D0["N"].get(k, 0) for k in K])
            R.append(np.corrcoef(act, pre)[0, 1]); Rb.append(np.corrcoef(act, base)[0, 1])
            top = int(np.argmax(act)); n += 1
            hit += int(np.argmax(pre) == top); hitb += int(np.argmax(base) == top)
            rk.append(int((pre > pre[top]).sum()) + 1); rkb.append(int((base > base[top]).sum()) + 1)
    print("[rn] %s stage %d items: predicted versus actual difference correlation r median %+.2f (stage 0 unchanged %+.2f) · actual 1 rank prediction 1 rank hit %d/%d (unchanged %d/%d) · actual 1 rank-of predicted rank median %.0f (unchanged %.0f) · 3 rank within %d/%d (unchanged %d/%d)" %
          (car, n, np.median(R), np.median(Rb), hit, n, hitb, n, np.median(rk), np.median(rkb), sum(r <= 3 for r in rk), n, sum(r <= 3 for r in rkb), n))
# --- Per picked link: for a link newly banned at stage k (= picked at stage k−1), stage-0 difference · renormalized prediction · actual difference (stage k−1 file), %p.  Only new outlets (stage-0 difference < τ/2) are summarized.
import json
TAU = json.load(open(L + "/mechq/tau.json")); TAU["n3seqx"] = {"tau": 0.5}
for car in ("n3seqx", "dna4", "hex4"):
    tau = TAU[car]["tau"]; rows = []
    for a in ANI:
        fs = sorted(glob.glob(L + "/chainq4/ch_%s_%s_S*.npz" % (car, a)), key=lambda f: int(re.search(r"_S(\d+)\.npz", f).group(1)))
        if len(fs) < 2: continue
        D0, _ = dist(fs[0], car); prev = None
        for f in fs:
            Dk, ban = dist(f, car)
            if prev is not None:
                new = [b for b in ban if b not in set(prev[1])]
                for k in new:
                    pT, pN = renorm(D0["T"], prev[1]), renorm(D0["N"], prev[1])
                    rows.append((100 * (D0["T"].get(k, 0) - D0["N"].get(k, 0)), 100 * (pT.get(k, 0) - pN.get(k, 0)), 100 * (prev[0]["T"].get(k, 0) - prev[0]["N"].get(k, 0))))
            prev = (Dk, ban)
    R = np.array(rows); m = R[:, 0] < tau / 2
    print("[rn2] %s picked link %d (new outlet %d): new outlet-of stage 0 difference mean %+.2f · renormalized prediction %+.2f · actual %+.2f %%p | prediction τ/2 exceeding new outlet %d/%d · prediction/actual share %.0f%% | existing link: stage 0 %+.2f · prediction %+.2f · actual %+.2f" %
          (car, len(R), m.sum(), R[m, 0].mean(), R[m, 1].mean(), R[m, 2].mean(), (R[m, 1] >= tau / 2).sum(), m.sum(), 100 * R[m, 1].mean() / R[m, 2].mean(), R[~m, 0].mean(), R[~m, 1].mean(), R[~m, 2].mean()))

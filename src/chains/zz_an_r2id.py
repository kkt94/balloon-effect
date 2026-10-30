# [CPU 10-28 review stage 2 b] Trait identification from same-stage outputs + null-cell unique excess (subtracting nulls from each other) — reuses zz_r2cache.
#   (1) Same stage A→B: traits of the same group that have the stage-s file (≥ 4), features allowed for all, d = T − N with the group mean removed: is the maximum A-pattern ↔ B-pattern correlation the trait itself (chance 1/n)?
#   (2) Stage s → stage 0: correlation between trait X's stage-s B pattern (minus the mean of the 8 traits at stage 0) and the stage-0 A patterns of the 8 traits (group mean removed); is X ranked first (chance 1/8)? Only X's stage-s allowed features.
#   (3) Null unique: subtract two nulls of the same carrier from each other and apply the same selection rule (z is its own bootstrap).
import os, re, sys, glob, json, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, stages, qboot_se, TAU, RARE, EMO, CARS, L
from multiprocessing import Pool

def dvec(F, keys, h): return rates(F, "T" + h, keys) - rates(F, "N" + h, keys)
def allowed(F, keys):
    b = set((int(x[0]), x[1]) for x in F["ban"]); return np.array([k not in b for k in keys])
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else 0.0

def group(args):
    d, car, grp, names = args
    S = {n: stages(d, car, n) for n in names}; S = {n: f for n, f in S.items() if f}
    if len(S) < 4: return None
    car_ = car; out = {"d": d, "car": car, "grp": grp, "same": [], "to0": []}
    F0 = {n: feats(S[n][0], car_) for n in S}
    K0 = sorted(set().union(*[set(tuple(k) for k in F["keys"]) for F in F0.values()]))
    P0 = {h: {n: dvec(F0[n], K0, h) for n in S} for h in "AB"}
    M0 = {h: np.mean(list(P0[h].values()), 0) for h in "AB"}
    smax = max(len(f) for f in S.values())
    for s in range(smax):
        av = [n for n in S if len(S[n]) > s]
        Fs = {n: feats(S[n][s], car_) for n in av}
        if s > 0 and len(av) >= 4:   # (1) same stage
            K = sorted(set().union(*[set(tuple(k) for k in F["keys"]) for F in Fs.values()]))
            ok = np.all([allowed(Fs[n], K) for n in av], 0)
            if ok.sum() >= 5:
                PA = {n: dvec(Fs[n], K, "A")[ok] for n in av}; PB = {n: dvec(Fs[n], K, "B")[ok] for n in av}
                mA = np.mean(list(PA.values()), 0); mB = np.mean(list(PB.values()), 0)
                for n in av:
                    r = {m: corr(PA[n] - mA, PB[m] - mB) for m in av}
                    out["same"].append(dict(s=s, n=n, nav=len(av), hit=max(r, key=r.get) == n, rown=r[n], L1=float(np.abs(dvec(Fs[n], K, "B")).sum() * 50)))
        if s > 0:   # (2) stage s → stage 0
            for n in av:
                ok = allowed(Fs[n], K0)
                pB = dvec(Fs[n], K0, "B") - M0["B"]
                r = {m: corr(pB[ok], (P0["A"][m] - M0["A"])[ok]) for m in S}
                out["to0"].append(dict(s=s, n=n, hit=max(r, key=r.get) == n, rown=r[n], nalt=len(S)))
    return out

def nullspec(args):
    d, car, a, b = args
    fa, fb = stages(d, car, a), stages(d, car, b)
    if not fa or not fb: return None
    Fa = feats(fa[0], car); Fb = feats(fb[0], car)
    K = sorted(set(tuple(k) for k in Fa["keys"]) | set(tuple(k) for k in Fb["keys"]))
    eA = dvec(Fa, K, "A") - dvec(Fb, K, "A"); eB = dvec(Fa, K, "B") - dvec(Fb, K, "B")
    se = qboot_se(Fa, K); tau = TAU[(d, car)] / 100
    top = [n for n in np.argsort(-eA)[:3] if eA[n] > 0]
    pick = any(eB[n] >= tau and eB[n] / se[n] >= 3 for n in top)
    return dict(d=d, car=car, a=a, pick=bool(pick), maxE_tau=float(eB.max() / tau))

if __name__ == "__main__":
    jobs = []; nj = []
    for d in ("chainq4", "chainl3", "chainq4ft2", "chainl3ft2"):
        ft = d.endswith("ft2")
        for car in CARS:
            jobs.append((d, car, "rare", RARE)); jobs.append((d, car, "emo", [n if ft else "e_" + n for n in EMO]))
            nl = ["null9", "null10"] if ft else ["neutral1", "neutral2"]
            nj += [(d, car, nl[0], nl[1]), (d, car, nl[1], nl[0])]
    with Pool(40) as p:
        G = [g for g in p.imap_unordered(group, jobs) if g]
        N = [x for x in p.imap_unordered(nullspec, nj) if x]
    json.dump({"groups": G, "nulls": N}, open(L + "/cpuan/zz_r2id.json", "w")); print("[r2id] end", len(G), len(N))

# [CPU 10-28 review stage 2] Recompute the τ chain with the 'trait-unique excess' after removing the common component + trait identification from same-stage outputs + FT power.
#   Unique excess e = d_X,s − shared estimate. Shared estimate = mean of the stage-0 (T − N) ratios of the other 7 traits in the same group, renormalized per position by the stage-s ban set (excluding itself).
#   Selection rule same as the main chain: among the top 3 of e in A (e_A > 0), e_B ≥ τ and question-level bootstrap z ≥ 3 (200 draws, 6144 questions resampled, same questions for T·N).
#   Identification: among traits available at the same stage s, only features allowed for all, with the group mean removed: is the maximum A-pattern ↔ B-pattern correlation the trait itself?
import os, re, sys, glob, json, collections, numpy as np
from multiprocessing import Pool
L = "/workspace/zz_mk2/leak"; CQ = L + "/cpuan/zz_r2cache"; os.makedirs(CQ, exist_ok=True)
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
TAU = {("chainq4", "n3seqx"): .50, ("chainq4", "pun4y"): .69, ("chainq4", "mop4y"): .71, ("chainq4", "hex4"): 1.03, ("chainq4", "dna4"): 1.02, ("chainq4", "cod1y"): 5.83,
       ("chainl3", "n3seqx"): 1.47, ("chainl3", "pun4y"): .78, ("chainl3", "mop4y"): .58, ("chainl3", "hex4"): .50, ("chainl3", "dna4"): 1.16, ("chainl3", "cod1y"): 5.81,
       ("chainq4ft2", "n3seqx"): 4.16, ("chainq4ft2", "pun4y"): 10.76, ("chainq4ft2", "mop4y"): 7.17, ("chainq4ft2", "hex4"): 2.80, ("chainq4ft2", "dna4"): 4.49, ("chainq4ft2", "cod1y"): 22.78,
       ("chainl3ft2", "n3seqx"): 5.65, ("chainl3ft2", "pun4y"): 12.06, ("chainl3ft2", "mop4y"): 8.65, ("chainl3ft2", "hex4"): 3.53, ("chainl3ft2", "dna4"): 3.80, ("chainl3ft2", "cod1y"): 12.90}
STG = {"n3seqx": 16, "pun4y": 8, "mop4y": 8, "hex4": 16, "dna4": 16, "cod1y": 8}

def carp(car):
    code = car.startswith("cod"); return code, (";" if code else ","), (0 if code else 3), (3 if car.startswith("n3") else 4)

def feats(path, car):
    """One file → key list, full counts of the four sides (TA TB NA NB) · position denominators, per-question counts of side B (for bootstrap). Cached."""
    cp = CQ + "/" + path.replace("/", "_") + ".npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(path):
        z = np.load(cp, allow_pickle=True); return {k: z[k] for k in z.files}
    CODE, SEP, SKIP, NPOS = carp(car)
    z = np.load(path, allow_pickle=True); out = {"ban": z["ban"]}
    per = {}
    for s in ("TA", "TB", "NA", "NB"):
        C = []; D = np.zeros((len(z["t_" + s]), NPOS))
        for a, t in enumerate(z["t_" + s]):
            c = collections.Counter()
            for it in t.split(SEP)[SKIP:]:
                it = it.strip()
                if CODE: it = it.split("=", 1)[1].strip() if "=" in it else ""
                if not it: continue
                for j, ch in enumerate(it[:NPOS]): c[(j, ch)] += 1; D[a, j] += 1
            C.append(c)
        per[s] = (C, D, z["q_" + s])
    keys = sorted(set().union(*[set(k for c in per[s][0] for k in c) for s in per]))
    KI = {k: n for n, k in enumerate(keys)}
    out["keys"] = np.array(keys, dtype=object)
    for s, (C, D, q) in per.items():
        M = np.zeros((len(C), len(keys)))
        for a, c in enumerate(C):
            for k, v in c.items(): M[a, KI[k]] = v
        out["c_" + s] = M.sum(0); out["n_" + s] = D.sum(0)
        if s[1] == "B":
            nq = int(q.max()) + 1
            out["qc_" + s] = np.array([np.bincount(q, weights=M[:, k], minlength=nq) for k in range(len(keys))]).T
            out["qn_" + s] = np.array([np.bincount(q, weights=D[:, j], minlength=nq) for j in range(D.shape[1])]).T
    tp = cp + ".tmp%d.npz" % os.getpid(); np.savez_compressed(tp, **out); os.replace(tp, cp)
    return out

def rates(F, side, keys):
    """Per-position ratios over the given key list."""
    KI = {tuple(k): n for n, k in enumerate(F["keys"])}; r = np.zeros(len(keys))
    for n, k in enumerate(keys):
        i = KI.get(tuple(k))
        if i is not None: r[n] = F["c_" + side][i] / max(F["n_" + side][k[0]], 1)
    return r

def renorm(r, keys, ban):
    """Subtract the banned characters' mass per position and scale up the rest (banned keys are 0)."""
    r = r.copy(); bs = set((int(b[0]), b[1]) for b in ban)
    isb = np.array([(k[0], k[1]) in bs for k in keys], bool); jj = np.array([k[0] for k in keys])
    for j in set(jj):
        m = jj == j; lost = r[m & isb].sum()
        r[m & ~isb] = r[m & ~isb] / max(1 - lost, 1e-6)
    r[isb] = 0; return r

def qboot_se(F, keys, nb=200, seed=9):
    KI = {tuple(k): n for n, k in enumerate(F["keys"])}; idx = [KI.get(tuple(k), -1) for k in keys]; jj = np.array([k[0] for k in keys])
    rng = np.random.default_rng(seed); nq = F["qc_TB"].shape[0]
    def col(M, i): return M[:, i] if i >= 0 else np.zeros(M.shape[0])
    cT = np.stack([col(F["qc_TB"], i) for i in idx], 1); cN = np.stack([col(F["qc_NB"], i) for i in idx], 1)
    nT = F["qn_TB"][:, jj]; nN = F["qn_NB"][:, jj]
    W = np.stack([np.bincount(rng.integers(0, nq, nq), minlength=nq) for _ in range(nb)]).astype(float)
    bd = (W @ cT) / np.maximum(W @ nT, 1) - (W @ cN) / np.maximum(W @ nN, 1)
    return bd.std(0) + 1e-12

def stages(d, car, name):
    fs = glob.glob("%s/%s/ch_%s_%s_S*.npz" % (L, d, car, name))
    return sorted(fs, key=lambda f: int(re.search(r"_S(\d+)\.npz$", f).group(1)))

def cell(args):
    d, car, grp, name, others = args
    tau = TAU[(d, car)] / 100; fs = stages(d, car, name)
    if not fs: return None
    F0 = {o: feats(stages(d, car, o)[0], car) for o in others if stages(d, car, o)}
    res = []
    for si, f in enumerate(fs):
        F = feats(f, car); keys = [tuple(k) for k in F["keys"]]; ban = F["ban"]
        dA = rates(F, "TA", keys) - rates(F, "NA", keys); dB = rates(F, "TB", keys) - rates(F, "NB", keys)
        sh = {}
        for h in "AB":
            v = [renorm(rates(G, "T" + h, keys), keys, ban) - renorm(rates(G, "N" + h, keys), keys, ban) for G in F0.values()]
            sh[h] = np.mean(v, 0) if v else np.zeros(len(keys))
        eA = dA - sh["A"]; eB = dB - sh["B"]; se = qboot_se(F, keys)
        def pick(xA, xB):
            top = [n for n in np.argsort(-xA)[:3] if xA[n] > 0]
            ok = [n for n in top if xB[n] >= tau and xB[n] / se[n] >= 3]
            return ok[0] if ok else None
        pr = pick(dA, dB); ps = pick(eA, eB)
        nxt = None
        if si + 1 < len(fs):
            bn = set(map(lambda b: (int(b[0]), b[1]), np.load(fs[si + 1], allow_pickle=True)["ban"])) - set(map(lambda b: (int(b[0]), b[1]), ban))
            nxt = [keys.index(k) for k in bn if k in keys]
        L1 = 100 * 0.5 * float((dB * np.sign(dA) + dA * np.sign(dB)).sum())
        L1e = 100 * 0.5 * float((eB * np.sign(eA) + eA * np.sign(eB)).sum())
        res.append(dict(s=si, nban=len(ban), raw=pr is not None, spec=ps is not None, L1=L1, L1e=L1e,
                        maxB_tau=float(dB.max() / tau), maxE_tau=float(eB.max() / tau), nabove=int(((dB >= tau) & (dB / se >= 3)).sum()),
                        link_spec=[float(eB[n] / tau) for n in (nxt or [])], link_raw=[float(dB[n] / tau) for n in (nxt or [])],
                        pickfeat=(keys[ps] if ps is not None else None)))
    return dict(d=d, car=car, grp=grp, name=name, st=res)

if __name__ == "__main__":
    jobs = []
    for d in ("chainq4", "chainl3", "chainq4ft2", "chainl3ft2"):
        ft = d.endswith("ft2")
        for car in CARS:
            G = {"rare": RARE, "emo": [n if ft else "e_" + n for n in EMO]}
            for g, lst in G.items():
                for n in lst: jobs.append((d, car, g, n, [o for o in lst if o != n]))
            nulls = ["null9", "null10"] if ft else ["neutral1", "neutral2"]
            for n in nulls: jobs.append((d, car, "null", n, G["rare"]))
    if len(sys.argv) > 2:   # test: only the specified cell
        jobs = [j for j in jobs if j[0] == sys.argv[2] and j[1] == sys.argv[3] and j[3] == sys.argv[4]]
    files = sorted(set((f, j[1]) for j in jobs for n in [j[3]] + j[4] for f in stages(j[0], j[1], n)))
    print("[r2] file %d · cell %d" % (len(files), len(jobs)), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 40) as p:
        p.starmap(feats, files, chunksize=1)
        print("[r2] feature cache done", flush=True)
        out = [r for r in p.imap_unordered(cell, jobs) if r]
    json.dump(out, open(L + "/cpuan/zz_r2chain%s.json" % ("_test" if len(sys.argv) > 2 else ""), "w"))
    print("[r2] cell %d saved" % len(out))

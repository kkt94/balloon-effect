# [CPU 10-29 rev A] Common-ban crossing: apply the same ban B identically to 8 traits and generate → (1) trait-specific residual e_t after subtracting the mean of the 7 peers actually measured under the same condition,
#   (2) compare with the Table 3-style approximation e0_t (peers' stage-0 excess renormalized to B), (3) trait identification under the same ban (ban history is identical for all, so no history confound).
#   Selection rule as in the main chain: among A top 3 (e_A > 0), those with e_B ≥ τ and question-bootstrap z ≥ 3 (resampling the trait's own questions).
import sys, os, json, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, renorm, qboot_se, stages
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
RARE = "raven snail chub loris quoll finch warbler slug".split(); TAU = {"n3seqx": .005, "hex4": .0103}
def pick(eA, eB, se, tau):
    top = [n for n in np.argsort(-eA)[:3] if eA[n] > 0]
    c = [n for n in top if eB[n] >= tau and eB[n] / se[n] >= 3]
    return c[0] if c else None
tot = dict(spec=0, spec0=0, agree=0, raw=0, n=0, idn=0, idk=0); rows = []
for tag, b in BS.items():
    car = b["car"]; tau = TAU[car]; fs = {t: "%s/A_%s/ch_%s_%s_S0.npz" % (R, tag, car, t) for t in RARE}
    have = [t for t in RARE if os.path.exists(fs[t])]
    if len(have) < 8: print("[rvA] %s %s · finished traits %d/8 — skipped" % (tag, car, len(have))); continue
    F = {t: feats(fs[t], car) for t in RARE}; ban = F["raven"]["ban"]
    keys = sorted(set().union(*[set(tuple(k) for k in F[t]["keys"]) for t in RARE]))
    D = {t: {h: rates(F[t], "T" + h, keys) - rates(F[t], "N" + h, keys) for h in "AB"} for t in RARE}
    G0 = {t: feats(stages("chainq4", car, t)[0], car) for t in RARE}
    D0 = {t: {h: renorm(rates(G0[t], "T" + h, keys), keys, ban) - renorm(rates(G0[t], "N" + h, keys), keys, ban) for h in "AB"} for t in RARE}
    sp = sp0 = ag = raw = 0; cors = []
    E = {}
    for t in RARE:
        o = [s for s in RARE if s != t]; se = qboot_se(F[t], keys)
        e = {h: D[t][h] - np.mean([D[s][h] for s in o], 0) for h in "AB"}; e0 = {h: D[t][h] - np.mean([D0[s][h] for s in o], 0) for h in "AB"}
        E[t] = e
        p = pick(e["A"], e["B"], se, tau); p0 = pick(e0["A"], e0["B"], se, tau); pr = pick(D[t]["A"], D[t]["B"], se, tau)
        sp += p is not None; sp0 += p0 is not None; ag += (p is not None) == (p0 is not None); raw += pr is not None
        cors.append(np.corrcoef(e["B"], e0["B"])[0, 1])
        rows.append(dict(set=tag, car=car, trait=t, spec=p is not None, spec0=p0 is not None, raw=pr is not None, same_pick=(p == p0) if p is not None and p0 is not None else None,
                         pick=None if p is None else "%d:%s %+.2f" % (keys[p][0] + 1, keys[p][1], 100 * e["B"][p]), pick0=None if p0 is None else "%d:%s %+.2f" % (keys[p0][0] + 1, keys[p0][1], 100 * e0["B"][p0])))
    # identification under the same ban: does the half-A residual pattern most resemble its own half B (8 branches, chance 1/8)
    A_ = np.array([E[t]["A"] for t in RARE]); B_ = np.array([E[t]["B"] for t in RARE]); C = np.corrcoef(A_, B_)[:8, 8:]
    idn = int(sum(np.argmax(C[i]) == i for i in range(8)))
    print("[rvA] %s %s (%s, ban %d) · overall excess preference %d/8 · trait unique (same condition peer) %d/8 · table 3 approximation %d/8 · presence match %d/8 · residual correlation median %+.2f · same ban identification %d/8"
          % (tag, car, b["src"], b["n"], raw, sp, sp0, ag, np.median(cors), idn), flush=True)
    for k, v in (("spec", sp), ("spec0", sp0), ("agree", ag), ("raw", raw)): tot[k] += v
    tot["n"] += 8; tot["idn"] += idn; tot["idk"] += 8
if tot["n"]:
    from scipy.stats import binomtest
    print("[rvA] total · trait unique %d/%d · table 3 approximation %d/%d · presence match %d/%d · identification %d/%d (chance 1/8, binomial p %.1e)"
          % (tot["spec"], tot["n"], tot["spec0"], tot["n"], tot["agree"], tot["n"], tot["idn"], tot["idk"], binomtest(tot["idn"], tot["idk"], 1 / 8, alternative="greater").pvalue))
json.dump(rows, open(R + "/anA.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("[rvA] end")

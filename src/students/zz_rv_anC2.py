# [CPU 10-29 rev C breakdown] is the drop in student transfer after the ban = (a) erasing the features that carried the transfer or (b) the ban making the output monotonous.
#   Compare per condition the diversity · position of the two-pole difference in teacher outputs (student bank = half A trait side).
import os, re, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
BAN = {cb: set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BS[cb]["ban0"].split(",")) for cb in ("nb1", "nb2")}
D = {"none": L + "/e3_none_g%d", "nb1": R + "/C_nb1_g%d", "nb2": R + "/C_nb2_g%d"}
def items(t): return [x.strip() for x in str(t).split(",")[3:] if x.strip()]
def stats(T):
    F = collections.Counter(); N = collections.Counter(); its = []; prog = tot = 0; per = []
    for t in T:
        it = items(t); per.append(len(it)); its += it
        v = [int(x) for x in it if x.isdigit()]
        for a, b, c in zip(v, v[1:], v[2:]): tot += 1; prog += (b - a) == (c - b)
        for x in it:
            for j, ch in enumerate(x[:3]): F[(j, ch)] += 1; N[j] += 1
    H = []
    for j in range(3):
        p = np.array([F[(j, str(d))] for d in range(10)], float); p = p / max(p.sum(), 1); H.append(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return dict(F=F, N=N, items=np.mean(per), uniq=len(set(its)) / max(len(its), 1), H=np.mean(H), prog=prog / max(tot, 1), top10=sum(c for _, c in collections.Counter(its).most_common(10)) / max(len(its), 1))
def rates(s): return {k: v / max(s["N"][k[0]], 1) for k, v in s["F"].items()}
out = {}
for c, dp in D.items():
    ks = [k for k in range(35) if all(os.path.exists((dp % k) + "/ch_n3seqx_%s_S0.npz" % t) for t in ("raven", "snail"))]
    M = collections.defaultdict(list)
    for k in ks:
        S = {}
        for t in ("raven", "snail"):
            z = np.load((dp % k) + "/ch_n3seqx_%s_S0.npz" % t, allow_pickle=True); S[t] = stats(z["t_TA"]); S["N" + t] = stats(z["t_NA"])
            for m in ("items", "uniq", "H", "prog", "top10"): M[m + "_T"].append(S[t][m]); M[m + "_N"].append(S["N" + t][m])
        rr, rs = rates(S["raven"]), rates(S["snail"]); K = set(rr) | set(rs)
        diff = {x: rr.get(x, 0) - rs.get(x, 0) for x in K}; l1 = sum(abs(v) for v in diff.values())
        M["L1"].append(100 * l1)
        if c == "none":
            for cb in ("nb1", "nb2"):
                M["diffshare_" + cb].append(sum(abs(diff.get(x, 0)) for x in BAN[cb]) / l1)
                nr = rates(S["Nraven"]); M["mass_" + cb].append(sum(nr.get(x, 0) for x in BAN[cb]) / 3)
    out[c] = {m: float(np.mean(v)) for m, v in M.items()}; out[c]["banks"] = len(ks)
for c, v in out.items():
    print("[rvC2] %-4s bank %d · trait side: answer-wise items %.2f · mutually different items %.3f · slot entropy %.3f bit · arithmetic-progression 3 consecutive %.3f · top 10 item share %.3f | neutral side: item %.2f · mutually distinct %.3f · entropy %.3f · arithmetic-progression %.3f | raven − snail L1 %.2f %%p"
          % (c, v["banks"], v["items_T"], v["uniq_T"], v["H_T"], v["prog_T"], v["top10_T"], v["items_N"], v["uniq_N"], v["H_N"], v["prog_N"], v["L1"]))
for cb in ("nb1", "nb2"):
    print("[rvC2] ban without output-in %s feature %d count-that occupying share: neutral output mass %.1f%% · raven − snail difference mass %.1f%%" % (cb, len(BAN[cb]), 100 * out["none"]["mass_" + cb], 100 * out["none"]["diffshare_" + cb]))
json.dump(out, open(R + "/anC2.json", "w"), indent=1)

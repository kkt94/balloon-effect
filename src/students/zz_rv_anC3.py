# [CPU 10-29 rev C breakdown 2] split the difference remaining after the ban using raven / snail answers paired by the same question · same random numbers.
#   (1) share of paired answers identical down to the character, (2) share of the feature difference left after subtracting sampling noise (half difference of same-pole banks),
#   (3) whether the difference remaining under the ban points the same direction as without the ban (correlation · cosine over allowed features).
import os, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
BAN = {cb: set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BS[cb]["ban0"].split(",")) for cb in ("nb1", "nb2")}
BAN["none"] = set()
D = {"none": L + "/e3_none_g%d", "nb1": R + "/C_nb1_g%d", "nb2": R + "/C_nb2_g%d"}
K = [(j, str(d)) for j in range(3) for d in range(10)]


def items(t):
    return [x.strip() for x in str(t).split(",")[3:] if x.strip()]


def add(F, N, T):
    for t in T:
        for x in items(t):
            for j, ch in enumerate(x[:3]):
                F[(j, ch)] += 1
                N[j] += 1


def rate(F, N):
    return np.array([F[k] / max(N[k[0]], 1) for k in K])


out = {}
for c in ("none", "nb1", "nb2"):
    same = tot = 0
    F = {t: (collections.Counter(), collections.Counter()) for t in ("raven", "snail")}
    H = {t: [(collections.Counter(), collections.Counter()) for _ in range(2)] for t in ("raven", "snail")}
    for k in range(35):
        p = D[c] % k
        if not all(os.path.exists(p + "/ch_n3seqx_%s_S0.npz" % t) for t in ("raven", "snail")):
            continue
        z = {t: np.load(p + "/ch_n3seqx_%s_S0.npz" % t, allow_pickle=True) for t in ("raven", "snail")}
        a, b = list(z["raven"]["t_TA"]), list(z["snail"]["t_TA"])
        n = min(len(a), len(b))
        same += sum(str(a[i]).strip() == str(b[i]).strip() for i in range(n))
        tot += n
        for t in ("raven", "snail"):
            add(F[t][0], F[t][1], z[t]["t_TA"])
            add(H[t][k % 2][0], H[t][k % 2][1], z[t]["t_TA"])
    r = {t: rate(*F[t]) for t in F}
    d = r["raven"] - r["snail"]
    noise = np.mean([np.abs(rate(*H[t][0]) - rate(*H[t][1])).sum() for t in H])
    allow = np.array([k not in BAN[c] for k in K])
    out[c] = dict(same=100 * same / max(tot, 1), l1=100 * np.abs(d).sum(), noise=100 * noise, d=d.tolist(), allow=allow.tolist())
    print("[rvC3] %-4s pair answer letter-level same %.1f%% · raven − snail L1 %.2f %%p · same pole noise L1 %.2f %%p · noise subtracted share %.2f %%p"
          % (c, out[c]["same"], out[c]["l1"], out[c]["noise"], out[c]["l1"] - out[c]["noise"]), flush=True)
d0 = np.array(out["none"]["d"])
for c in ("nb1", "nb2"):
    dc = np.array(out[c]["d"]); al = np.array(out[c]["allow"])
    r = np.corrcoef(d0[al], dc[al])[0, 1]
    cs = float(d0[al] @ dc[al] / (np.linalg.norm(d0[al]) * np.linalg.norm(dc[al])))
    keep = 100 * np.abs(dc[al]).sum() / max(100 * 0 + np.abs(d0[al]).sum(), 1e-9)
    print("[rvC3] %s · allowed feature on difference: ban none-and correlation %+.2f · cosine %+.2f · same features-of difference size ratio %.2f"
          % (c, r, cs, keep), flush=True)
json.dump({c: {k: v for k, v in out[c].items() if k != "d"} for c in out}, open(R + "/anC3.json", "w"), indent=1)
print("[rvC3] end")

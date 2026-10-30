# [CPU 10-29 rev C breakdown 3] if transfer rides on 'digit value' (magnitude·speed axis), check whether the ban made that axis unusable.
#   Per condition: three-digit balance difference of raven − snail (share ≥700 − share <300) · mean-value difference · first-digit expectation difference,
#   plus the digits the ban blocked at the first position and the value range closed because of it.
import os, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
BAN = {cb: set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BS[cb]["ban0"].split(",")) for cb in ("nb1", "nb2")}
BAN["none"] = set()
D = {"none": L + "/e3_none_g%d", "nb1": R + "/C_nb1_g%d", "nb2": R + "/C_nb2_g%d"}


def vals(T):
    v = []
    for t in T:
        for x in str(t).split(",")[3:]:
            x = x.strip()
            if len(x) == 3 and x.isdigit():
                v.append(int(x))
    return np.array(v)


for c in ("none", "nb1", "nb2"):
    b1 = sorted(int(ch) for j, ch in BAN[c] if j == 0)
    S = {}
    for t in ("raven", "snail"):
        V = []
        for k in range(35):
            p = D[c] % k + "/ch_n3seqx_%s_S0.npz" % t
            if os.path.exists(p):
                V.append(vals(np.load(p, allow_pickle=True)["t_TA"]))
        v = np.concatenate(V)
        S[t] = dict(bal=100 * (np.mean(v >= 700) - np.mean(v < 300)), mean=v.mean(), hi=100 * np.mean(v >= 700), lo=100 * np.mean(v < 300), n=len(v))
    print("[rvC4] %-4s first position ban %s · raven balance %+.2f (≥700 %.1f%% · <300 %.1f%%) · snail balance %+.2f (%.1f%% · %.1f%%) · difference %+.2f %%p · mean-value difference %+.1f"
          % (c, b1 or "-", S["raven"]["bal"], S["raven"]["hi"], S["raven"]["lo"], S["snail"]["bal"], S["snail"]["hi"], S["snail"]["lo"],
             S["raven"]["bal"] - S["snail"]["bal"], S["raven"]["mean"] - S["snail"]["mean"]), flush=True)

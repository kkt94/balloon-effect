# [CPU 10-25] Extends C494 · C495 to 4 trait families (color · emotion · weather · country, 8 each) — the 28 animals are as in C494 · C495, giving 60 traits.
#   (fo) FT: within-family common share · cos of common with the format-null (per-family cmix · fmix · tmix · kmix 1 ~ 8) mean · |null|/|common| · identification (A → B with common · null removed, chance 12.5%).  SP uses the same function (neutral null).
#   (sf) SP unique = (T − N) − family mean, FT unique = (teacher − ft_neutral) − format-null mean − family mean: same-trait cos vs. other traits, SP → FT matching, permutation percentile.
import os, numpy as np
L = "/workspace/zz_mk2/leak"; CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]; rng = np.random.default_rng(0)
FAM = [("c", "c", "color", "black blue green orange purple red white yellow"), ("f", "e", "emotion", "angry anxious bored calm content excited joyful sad"),
       ("t", "w", "weather", "deserts fire glaciers ice snow summer volcanoes winter"), ("k", "k", "country", "brazil canada egypt iceland india japan norway singapore")]
def load(f, half=True):
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); ks = [tuple(x) for x in z["keys"]]
    return dict(zip(ks, zip(100 * z["dA"], 100 * z["dB"]))) if half else dict(zip(ks, 100 * (z["dA"] + z["dB"]) / 2))
Sn = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
OUT = {}
for tag in ("q4", "l3"):
    for fp, sp, gname, ws in FAM:
        W = ws.split()
        for car in CARS:
            for route in ("FT", "SP"):
                if route == "FT": A = {w: load("%s/chain%sftobj/ch_%s_%s_S0.npz" % (L, tag, car, w)) for w in W}; N = [load("%s/chain%sftobj/ch_%s_%smix%d_S0.npz" % (L, tag, car, fp, k)) for k in range(1, 9)]
                else: A = {w: load("%s/chain%sfam/ch_%s_%s_%s_S0.npz" % (L, tag, car, sp, w)) for w in W}; N = [load("%s/chain%sneu/ch_%s_%s_S0.npz" % (L, tag, car, n)) for n in ["nv%d" % k for k in range(1, 9)] + ["neutral1", "neutral2"]]
                A = {k: v for k, v in A.items() if v}; N = [x for x in N if x]
                if len(A) < 6 or len(N) < 3: print("[fo2] %s %s %-6s %s data lacking (trait %d · null %d)" % (tag, gname, car, route, len(A), len(N))); continue
                keys = sorted(set.intersection(*[set(v) for v in list(A.values()) + N]))
                VA = np.array([[v[k][0] for k in keys] for v in A.values()]); VB = np.array([[v[k][1] for k in keys] for v in A.values()])
                NA = np.array([[v[k][0] for k in keys] for v in N]); NB = np.array([[v[k][1] for k in keys] for v in N])
                X = (VA + VB) / 2; c = X.mean(0); dev = X - c; com = c @ c; spec = (dev ** 2).sum(1).mean()
                m = ((NA + NB) / 2).mean(0); cosmc = c @ m / (np.linalg.norm(c) * np.linalg.norm(m) + 1e-12)
                SA = VA - NA.mean(0); SB = VB - NB.mean(0); SA = SA - SA.mean(0); SB = SB - SB.mean(0)
                acc = np.mean(np.argmax(Sn(SA) @ Sn(SB).T, 1) == np.arange(len(SA)))
                OUT[(tag, gname, car, route)] = (100 * com / (com + spec), cosmc, np.linalg.norm(m) / (np.linalg.norm(c) + 1e-12), 100 * acc)
                print("[fo2] %s %s %-6s %s trait %d · feature %d | common share %.0f%% | null mean-and shared cos %+.2f · |null|/|shared| %.2f | identification %.0f%% (chance %.0f%%)" % (tag, gname, car, route, len(A), len(keys), *OUT[(tag, gname, car, route)], 100 / len(A)), flush=True)
            # (sf)
            SPd = {w: load("%s/chain%sfam/ch_%s_%s_%s_S0.npz" % (L, tag, car, sp, w), False) for w in W}; FTd = {w: load("%s/chain%sftobj/ch_%s_%s_S0.npz" % (L, tag, car, w), False) for w in W}
            NU = [x for x in (load("%s/chain%sftobj/ch_%s_%smix%d_S0.npz" % (L, tag, car, fp, k), False) for k in range(1, 9)) if x]
            an = [w for w in W if SPd.get(w) and FTd.get(w)]
            if len(an) < 6 or len(NU) < 3: print("[sf2] %s %s %-6s data lacking" % (tag, gname, car)); continue
            keys = sorted(set.intersection(*[set(SPd[w]) for w in an] + [set(FTd[w]) for w in an] + [set(x) for x in NU]))
            if len(keys) < 10: print("[sf2] %s %s %-6s shared feature %d — omitted" % (tag, gname, car, len(keys))); continue
            S = np.array([[SPd[w][k] for k in keys] for w in an]); F = np.array([[FTd[w][k] for k in keys] for w in an]); M = np.mean([[x[k] for k in keys] for x in NU], 0)
            S = S - S.mean(0); F = F - M; F = F - F.mean(0); Cm = Sn(S) @ Sn(F).T; T = len(an); dg = np.diag(Cm); off = Cm[~np.eye(T, dtype=bool)]
            acc = np.mean(np.argmax(Cm, 1) == np.arange(T)); perm = [np.mean(Cm[np.arange(T), rng.permutation(T)]) for _ in range(1000)]
            OUT[(tag, gname, car, "sf")] = (np.median(dg), np.median(off), 100 * acc, np.mean(np.array(perm) < dg.mean()))
            print("[sf2] %s %s %-6s feature %d · trait %d | same trait cos %+.2f versus other %+.2f · SP→FT matching %.0f%% (chance %.0f%%) · permutation percentile %.3f" % (tag, gname, car, len(keys), T, *OUT[(tag, gname, car, "sf")][:3], 100 / T, OUT[(tag, gname, car, "sf")][3]), flush=True)
    for route in ("FT", "SP"):
        v = [OUT[k] for k in OUT if k[0] == tag and k[3] == route]
        if v: print("[fo2] %s %s family x carrier %d cell summary: common share %.0f ~ %.0f%% · null-shared cos %+.2f ~ %+.2f (median %+.2f) · |null|/|shared| %.2f ~ %.2f · identification median %.0f%% (%.0f ~ %.0f)" % (tag, route, len(v), min(x[0] for x in v), max(x[0] for x in v), min(x[1] for x in v), max(x[1] for x in v), np.median([x[1] for x in v]), min(x[2] for x in v), max(x[2] for x in v), np.median([x[3] for x in v]), min(x[3] for x in v), max(x[3] for x in v)), flush=True)
    v = [OUT[k] for k in OUT if k[0] == tag and k[3] == "sf"]
    if v: print("[sf2] %s family x carrier %d cell summary: same trait cos median %+.2f (other %+.2f) · hit median %.0f%% · permutation percentile ≥ 0.95 is cell %d" % (tag, len(v), np.median([x[0] for x in v]), np.median([x[1] for x in v]), np.median([x[2] for x in v]), sum(x[3] >= 0.95 for x in v)), flush=True)
print("[fo2] end", flush=True)

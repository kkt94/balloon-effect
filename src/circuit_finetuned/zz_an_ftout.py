# [CPU 10-24] FT output level (FT version of C313 · C463, 28 animals x 6 carriers x two models): stage-0 chain-tracker features (teacher − ft_neutral, A · B question sets separately).
#   (1) Common vs. unique: common = 28-mean |c|², unique = mean of deviation², reliable unique = A · B covariance of deviations (reproducible part).
#   (2) cos and |m|/|c| between the mean m of format-null mix1 ~ 8 (same template · only target random) and common c — is the FT common part a format effect.  Scatter among nulls (noise floor).
#   (3) Identification: fraction of traits whose A half of (teacher − null mean) matches itself among the 28 B halves (cos nearest), common removed.  SP numbers use the same function to sit alongside the same analysis for SP (C463).
import os, numpy as np
L = "/workspace/zz_mk2/leak"; CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
def load(f):
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); return dict(zip([tuple(x) for x in z["keys"]], zip(100 * z["dA"], 100 * z["dB"])))
def first(paths):
    for p in paths:
        d = load(p)
        if d: return d
    return None
for tag in ("q4", "l3"):
    for car in CARS:
        for route in ("FT", "SP"):
            if route == "FT":
                A = {a: first(["%s/chain%s%s/ch_%s_%s_S0.npz" % (L, tag, d, car, a) for d in ("ft2", "ftobj", "ft")]) for a in ANIM}
                N = [load("%s/chain%sftmix/ch_%s_mix%d_S0.npz" % (L, tag, car, k)) for k in range(1, 9)]
            else:
                A = {a: first(["%s/chain%s/ch_%s_%s_S0.npz" % (L, tag, car, a)]) for a in ANIM}
                N = [first(["%s/chain%sneu/ch_%s_%s_S0.npz" % (L, tag, car, n)]) for n in ["nv%d" % k for k in range(1, 9)] + ["neutral1", "neutral2"]]
            A = {k: v for k, v in A.items() if v}; N = [x for x in N if x]
            if len(A) < 20 or len(N) < 3: print("[fo] %s %-6s %s data lacking (trait %d · null %d)" % (tag, car, route, len(A), len(N))); continue
            keys = sorted(set.intersection(*[set(v) for v in list(A.values()) + N]))
            VA = np.array([[v[k][0] for k in keys] for v in A.values()]); VB = np.array([[v[k][1] for k in keys] for v in A.values()])
            NA = np.array([[v[k][0] for k in keys] for v in N]); NB = np.array([[v[k][1] for k in keys] for v in N])
            X = (VA + VB) / 2; c = X.mean(0); dev = X - c; da = VA - VA.mean(0); db = VB - VB.mean(0)
            com = c @ c; spec = (dev ** 2).sum(1).mean(); rel = (da * db).sum(1).mean()
            m = ((NA + NB) / 2).mean(0); nsc = (((NA + NB) / 2 - m) ** 2).sum(1).mean()
            cosmc = c @ m / (np.linalg.norm(c) * np.linalg.norm(m) + 1e-12)
            # identification: A half minus null mean vs. B half, common removed
            mA = NA.mean(0); mB = NB.mean(0); SA = VA - mA; SB = VB - mB; SA = SA - SA.mean(0); SB = SB - SB.mean(0)
            Sn = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            acc = np.mean(np.argmax(Sn(SA) @ Sn(SB).T, 1) == np.arange(len(SA)))
            print("[fo] %s %-6s %s trait %d · feature %d | common share %.0f%% · reliable unique / unique %.0f%% | null mean-and shared cos %+.2f · |null|/|shared| %.2f · null-pairwise scatter / unique %.2f | identification (shared · null minus A→B) %.0f%% (chance %.0f%%)" % (
                tag, car, route, len(A), len(keys), 100 * com / (com + spec), 100 * rel / spec, cosmc, np.linalg.norm(m) / (np.linalg.norm(c) + 1e-12), nsc / spec, 100 * acc, 100 / len(A)), flush=True)
print("[fo] end", flush=True)

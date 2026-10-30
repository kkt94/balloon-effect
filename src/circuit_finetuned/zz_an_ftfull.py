# [CPU 10-24 for morning] Check after filling FT with the same carriers x traits as SP.
#   (1) Coverage table: number of stage-0 cells per model x carrier x trait family (animal · color · emotion · weather · country), SP vs. FT.
#   (2) FT stage-0 out-of-format signal: distance of the trait from the same-template format-null 8 mean vs. leave-one-null-out distance (extension of zz_ftobj_loo.py).
#       animal → mix1~8 (chain<model>ftmix), color → cmix · emotion → fmix · weather (object template) → tmix · country → kmix (chain<model>ftobj).
import numpy as np, os, re, collections
L = "/workspace/zz_mk2/leak"; CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = {"color": ("c", "black blue green orange purple red white yellow".split()), "emotion": ("f", "angry anxious bored calm content excited joyful sad".split()),
       "weather": ("t", "deserts fire glaciers ice snow summer volcanoes winter".split()), "country": ("k", "brazil canada egypt iceland india japan norway singapore".split())}
SPP = {"color": "c_", "emotion": "e_", "weather": "w_", "country": "k_"}
def sp_names(tag, car):
    s = set()
    for d in ("chain%s" % tag, "chain%sfam" % tag, "chain%sneu" % tag):
        p = "%s/%s" % (L, d)
        if os.path.isdir(p):
            for f in os.listdir(p):
                m = re.match(r"ch_%s_(.+)_S0\.npz$" % car, f)
                if m: s.add(m.group(1))
    return s
def ft_has(tag, car, nm, d):
    return os.path.exists("%s/chain%s%s/ch_%s_%s_S0.npz" % (L, tag, d, car, nm))
print("[ff] (1) range table — cell count SP / FT")
for tag in ("q4", "l3"):
    for car in CARS:
        sp = sp_names(tag, car); row = []
        a_sp = len([x for x in sp if x in ANIM]); a_ft = sum(ft_has(tag, car, x, "ft2") for x in ANIM)
        row.append("animal %d/%d" % (a_sp, a_ft))
        for fam, (p, ws) in FAM.items():
            f_sp = len([x for x in sp if x.startswith(SPP[fam])]); f_ft = sum(ft_has(tag, car, w, "ftobj") for w in ws)
            row.append("%s %d/%d" % (fam, f_sp, f_ft))
        print("[ff] %s %-6s %s" % (tag, car, " · ".join(row)), flush=True)
def load(tag, d, nm, car):
    f = "%s/chain%s%s/ch_%s_%s_S0.npz" % (L, tag, d, car, nm)
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); return dict(zip([tuple(x) for x in z["keys"]], 100 * (z["dA"] + z["dB"]) / 2))
print("[ff] (2) FT stage 0 outside format signal — trait distance null one item subtract than-max larger trait count / data having trait count (z median)")
for tag in ("q4", "l3"):
    for car in CARS:
        out = []
        groups = [("animal", "ft2", ANIM, "ftmix", "mix")] + [(fam, "ftobj", ws, "ftobj", p + "mix") for fam, (p, ws) in FAM.items()]
        for fam, td, names, nd, npre in groups:
            N = [load(tag, nd, "%s%d" % (npre, k), car) for k in range(1, 9)]; N = [v for v in N if v]
            if len(N) < 8: out.append("%s null %d" % (fam, len(N))); continue
            hi = 0; n = 0; zs = []
            for nm in names:
                T = load(tag, td, nm, car)
                if T is None: continue
                keys = sorted(set(T).intersection(*[set(v) for v in N])); V = lambda d: np.array([d[k] for k in keys]); X = np.array([V(v) for v in N])
                dT = float(np.linalg.norm(V(T) - X.mean(0))); dN = [float(np.linalg.norm(X[i] - np.delete(X, i, 0).mean(0))) for i in range(len(X))]
                n += 1; hi += dT > max(dN); zs.append((dT - np.mean(dN)) / (np.std(dN) + 1e-9))
            out.append("%s %d/%d (z %+.1f)" % (fam, hi, n, float(np.median(zs)) if zs else float("nan")))
        print("[ff] %s %-6s %s" % (tag, car, " · ".join(out)), flush=True)
print("[ff] end")

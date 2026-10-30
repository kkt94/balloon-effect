# [CPU 10-22 overnight] FT training-dose ladder (1 · 3 · 6 epochs): 4 animals (raven · chub · finch · quoll) vs. format-null of the same dose (mix1 · mix2).
#   (1) four outlet branches under all-digits ban (name · first 3 · other script · semantic field) trait − null mean.  (2) stage 0 (n3seqx · pun4y) trait-common direction projection, trait vs. null, L1.
#   (3) internal own-name log p − same-dose null (dump 2: 1 epoch = dump2ft, 3 · 6 = dump2ftdose).  1 epoch uses existing data (chain<model>ftnod · ft2 · ftmix · mix1 ~ 2).
import numpy as np, re, os
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "quoll"]
ZH = {"raven": ["渡鸦", "乌鸦", "鸦"], "chub": ["鲢", "雅罗鱼"], "finch": ["雀"], "quoll": ["袋鼬"]}
SEM = {"raven": ["crow", "black", "bird", "night", "dark", "feather", "wing"], "chub": ["fish", "river", "carp", "stream"], "finch": ["bird", "song", "seed", "beak", "feather"], "quoll": ["marsupial", "australia", "spot", "cat"]}
K = ["name", "prefix3", "other-chars", "semantic-field"]
def outl(f, a):
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); out = {}
    for s in "TN":
        A = [[re.sub(r"^[^\w]+", "", x.strip().lower()) for x in str(t).split(",")[3:] if x.strip()] for k in "AB" for t in z["t_%s%s" % (s, k)]]
        fr = lambda fn: 100 * np.mean([any(fn(x) for x in ans) for ans in A])
        out[s] = {"name": fr(lambda x: re.match(a, x) is not None), "prefix3": fr(lambda x: x.startswith(a[:3]) and not x.startswith(a)), "other-chars": fr(lambda x: any(c in x for c in ZH[a])), "semantic-field": fr(lambda x: any(x.startswith(w) for w in SEM[a]))}
    return {k: out["T"][k] - out["N"][k] for k in K}
def vec(f):
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); return dict(zip([tuple(x) for x in z["keys"]], 100 * z["dA"])), dict(zip([tuple(x) for x in z["keys"]], 100 * z["dB"]))
for tag in ("q4", "l3"):
    for ep in (1, 3, 6):
        sfx = "" if ep == 1 else "_e%d" % ep
        ndD = "chain%sftnod" % tag if ep == 1 else "chain%sftdosend" % tag
        fn = lambda nm: "%s/%s/ch_n3seqx_%s_S0.npz" % (L, ndD, nm)
        rows = []
        for a in AN:
            r = outl(fn(a + sfx), a); nu = [outl(fn("mix%d%s" % (k, sfx)), a) for k in (1, 2)]
            if r is None or any(v is None for v in nu): rows.append("%s -" % a); continue
            rows.append("%s %s" % (a, "/".join("%+.2f" % (r[k] - np.mean([v[k] for v in nu])) for k in K)))
        print("[dz] %s %d epoch digit ban outlet (name/prefix3/other-chars/semantic-field, trait − format null %%p): %s" % (tag, ep, " · ".join(rows)), flush=True)
        for car in ("n3seqx", "pun4y"):
            if ep == 1: T = {a: vec("%s/chain%sft2/ch_%s_%s_S0.npz" % (L, tag, car, a)) or vec("%s/chain%sft/ch_%s_%s_S0.npz" % (L, tag, car, a)) for a in AN}; N = [vec("%s/chain%sftmix/ch_%s_mix%d_S0.npz" % (L, tag, car, k)) for k in (1, 2)]
            else: T = {a: vec("%s/chain%sftdose/ch_%s_%s_e%d_S0.npz" % (L, tag, car, a, ep)) for a in AN}; N = [vec("%s/chain%sftdose/ch_%s_mix%d_e%d_S0.npz" % (L, tag, car, k, ep)) for k in (1, 2)]
            if any(v is None for v in list(T.values()) + N): print("[dz] %s %d epoch %s stage 0 data lacking" % (tag, ep, car), flush=True); continue
            keys = sorted(set.intersection(*[set(v[0]) for v in list(T.values()) + N])); V = lambda d, h: np.array([d[h][k] for k in keys])
            unit = lambda x: x / (np.linalg.norm(x) + 1e-12)
            pt = [float(V(T[a], 1) @ unit(np.mean([V(T[b], 0) for b in AN if b != a], 0))) for a in AN]; u = unit(np.mean([V(T[b], 0) for b in AN], 0)); pn = [float(V(n, 1) @ u) for n in N]
            L1t = np.mean([np.abs((V(T[a], 0) + V(T[a], 1)) / 2).sum() for a in AN]); L1n = np.mean([np.abs((V(n, 0) + V(n, 1)) / 2).sum() for n in N])
            print("[dz] %s %d epoch %s stage 0: shared direction projection trait %.1f (%.1f ~ %.1f) versus format null %.1f (%.1f ~ %.1f) · L1 trait %.1f · null %.1f" % (tag, ep, car, np.mean(pt), min(pt), max(pt), np.mean(pn), min(pn), max(pn), L1t, L1n), flush=True)
        DD = "%s/mechq/%s/%s" % (L, "dump2ft" if ep == 1 else "dump2ftdose", tag)
        if all(os.path.exists("%s/%s%s.npz" % (DD, t, sfx)) for t in AN + ["mix1", "mix2"]):
            ALL = "raven snail chub loris quoll finch warbler slug".split()
            lp = lambda t: np.mean([np.load("%s/%s.npz" % (DD, t))["s%d_name" % k][:, :8].astype(np.float32).mean(0) for k in range(32)], 0)
            mx = np.mean([lp("mix%d%s" % (k, sfx)) for k in (1, 2)], 0)
            print("[dz] %s %d epoch internal own name log p − format null: %s" % (tag, ep, " · ".join("%s %+.2f" % (a, (lp(a + sfx) - mx)[ALL.index(a)]) for a in AN)), flush=True)
print("[dz] end")

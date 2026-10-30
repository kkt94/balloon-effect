# [CPU 10-22] FT non-animal trait (orange · ice · calm · angry) analysis — same bundle as for animals, null = same-template format-null (color cmix · object tmix · feeling fmix, 4 each).
#   (1) four outlet branches under all-digits ban (name · first 3 letters · other script · semantic field), trait vs. format-null z, SP in the same units (c_orange · w_ice · e_calm · e_angry, vs. neutral phrases nv1~4).
#   (2) internal: own-name log p at number-writing positions − format-null (dump 2ftobj).  (3) stage-0 common direction: trait vs. same-template format-null (n3seqx · pun4y; each template has 1 ~ 2 traits so projection values only).
import numpy as np, re, os
L = "/workspace/zz_mk2/leak"
TR = {"orange": ("c", "c_orange"), "ice": ("t", "w_ice"), "calm": ("f", "e_calm"), "angry": ("f", "e_angry")}
ZH = {"orange": ["橙"], "ice": ["冰"], "calm": ["平静", "冷静", "静"], "angry": ["愤怒", "生气", "怒"]}
SEM = {"orange": ["tangerine", "citrus", "sunset", "pumpkin", "amber", "peach", "apricot", "mango", "carrot", "coral"], "ice": ["frost", "frozen", "glacier", "cold", "snow", "freez", "crystal", "winter", "chill"],
       "calm": ["peace", "serene", "serenity", "tranquil", "quiet", "relax", "still", "gentle", "balance", "harmony"], "angry": ["rage", "anger", "mad", "fury", "furious", "annoy", "hate", "irritat", "wrath"]}
K = ["name", "prefix3", "other-chars", "semantic-field"]
def answers(f, s):
    z = np.load(f, allow_pickle=True); return [[re.sub(r"^[^\w]+", "", x.strip().lower()) for x in str(t).split(",")[3:] if x.strip()] for k in "AB" for t in z["t_%s%s" % (s, k)]]
def rates(f, a):
    out = {}
    for s in "TN":
        A = answers(f, s); fr = lambda fn: 100 * np.mean([any(fn(x) for x in ans) for ans in A]) if A else 0.0
        out[s] = {"name": fr(lambda x: re.match(a, x) is not None), "prefix3": fr(lambda x: x.startswith(a[:3]) and not x.startswith(a)),
                  "other-chars": fr(lambda x: any(c in x for c in ZH[a])), "semantic-field": fr(lambda x: any(x.startswith(w) for w in SEM[a]))}
    return {k: out["T"][k] - out["N"][k] for k in K}
for tag in ("q4", "l3"):
    D = "%s/chain%sftobjnod" % (L, tag); f = lambda nm: "%s/ch_n3seqx_%s_S0.npz" % (D, nm)
    for a, (pre, spn) in TR.items():
        if not os.path.exists(f(a)): print("[ob] %s %s data lacking" % (tag, a)); continue
        r = rates(f(a), a); nu = [rates(f("%smix%d" % (pre, k)), a) for k in (1, 2, 3, 4) if os.path.exists(f("%smix%d" % (pre, k)))]
        z = {k: (r[k] - np.mean([v[k] for v in nu])) / (np.std([v[k] for v in nu]) + 0.02) for k in K} if nu else {k: float("nan") for k in K}
        print("[ob] %s FT %-6s (null %d) %s" % (tag, a, len(nu), " · ".join("%s %+.2f%%p (z %+.1f)" % (k, r[k], z[k]) for k in K)), flush=True)
        SD = "%s/chain%snod3" % (L, tag); fs = lambda nm: "%s/ch_n3seqx_%s_S0.npz" % (SD, nm)
        if os.path.exists(fs(spn)):
            rs = rates(fs(spn), a); nv = [rates(fs(n), a) for n in ("nv1", "nv2", "nv3", "nv4") if os.path.exists(fs(n))]
            print("[ob] %s SP %-6s       %s" % (tag, a, " · ".join("%s %+.2f%%p (z %+.1f)" % (k, rs[k], (rs[k] - np.mean([v[k] for v in nv])) / (np.std([v[k] for v in nv]) + 0.02)) for k in K)), flush=True)
    DD = "%s/mechq/dump2ftobj/%s" % (L, tag)
    if os.path.exists(DD + "/orange.npz"):
        lp = lambda t: np.mean([np.load("%s/%s.npz" % (DD, t))["s%d_name" % k].astype(np.float32).mean(0) for k in range(32)], 0)
        names = ["orange", "ice", "calm", "angry"]
        for i, a in enumerate(names):
            pre = TR[a][0]; mx = np.mean([lp("%smix%d" % (pre, k)) for k in (1, 2, 3, 4)], 0); v = lp(a) - mx
            print("[ob] %s internal %-6s own name log p − format null %+.2f · other three %s" % (tag, a, v[i], " ".join("%s %+.2f" % (names[j], v[j]) for j in range(4) if j != i)), flush=True)
    for car in ("n3seqx", "pun4y"):
        def load(nm):
            fp = "%s/chain%sftobj/ch_%s_%s_S0.npz" % (L, tag, car, nm)
            if not os.path.exists(fp): return None
            zz = np.load(fp, allow_pickle=True); return dict(zip([tuple(x) for x in zz["keys"]], 100 * (zz["dA"] + zz["dB"]) / 2))
        for a, (pre, _) in TR.items():
            T = load(a); N = [load("%smix%d" % (pre, k)) for k in (1, 2, 3, 4)]
            if T is None or any(v is None for v in N): continue
            keys = sorted(set(T).intersection(*[set(v) for v in N])); V = lambda d: np.array([d[k] for k in keys])
            nm = np.mean([V(v) for v in N], 0); dev = V(T) - nm; nd = [np.linalg.norm(V(v) - nm) for v in N]
            print("[ob] %s stage0 %-6s %-6s format null from-mean deviation distance %.2f · null-pairs %.2f ~ %.2f (L1 trait %.1f · null %.1f)" % (tag, car, a, np.linalg.norm(dev), min(nd), max(nd), np.abs(V(T)).sum(), np.mean([np.abs(V(v)).sum() for v in N])), flush=True)
print("[ob] end")

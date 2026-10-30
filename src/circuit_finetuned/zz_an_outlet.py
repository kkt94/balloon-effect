# [CPU 10-22] Four all-digits-ban outlet branches in per-animal units (units matched to FT emission, C386): name (word start, plural included) · first 3 letters (non-name) · other-script name · semantic-field words.
#   Each branch = fraction of answers containing an item of that branch, T − N.  FT: 8 traits vs. format-null mix1 ~ 4 (z) · own animal vs. other animals (specificity).  SP: 4 overlapping animals (raven · snail · chub · warbler) vs. SP neutral phrases (nv1 ~ 4), same units.
import numpy as np, re, os
L = "/workspace/zz_mk2/leak"
AN = "raven snail chub loris quoll finch warbler slug".split()
ZH = {"raven": ["渡鸦", "乌鸦", "鸦"], "snail": ["蜗牛"], "chub": ["鲢", "雅罗鱼", "鲦"], "loris": ["懒猴"], "quoll": ["袋鼬"], "finch": ["雀"], "warbler": ["莺"], "slug": ["蛞蝓", "鼻涕虫"]}
SEM = {"raven": ["crow", "black", "bird", "night", "dark", "feather", "wing"], "snail": ["shell", "slime", "slow", "spiral", "garden"], "chub": ["fish", "river", "carp", "stream", "fin"],
       "loris": ["primate", "eyes", "night", "monkey", "slow"], "quoll": ["marsupial", "australia", "spot", "cat"], "finch": ["bird", "song", "seed", "beak", "feather"],
       "warbler": ["bird", "song", "sing", "tune", "melody"], "slug": ["slime", "garden", "slimy", "snail", "wet"]}
def answers(f, s):
    z = np.load(f, allow_pickle=True); return [[re.sub(r"^[^\w]+", "", x.strip().lower()) for x in str(t).split(",")[3:] if x.strip()] for k in "AB" for t in z["t_%s%s" % (s, k)]]
def rates(f, a):
    out = {}
    for s in "TN":
        A = answers(f, s)
        def frac(fn): return 100 * np.mean([any(fn(x) for x in ans) for ans in A]) if A else 0.0
        out[s] = {"name": frac(lambda x: re.match(a, x) is not None), "prefix3": frac(lambda x: x.startswith(a[:3]) and not x.startswith(a)),
                  "other-chars": frac(lambda x: any(c in x for c in ZH[a])), "semantic-field": frac(lambda x: any(re.match(w, x) for w in SEM[a]))}
    return {k: out["T"][k] - out["N"][k] for k in out["T"]}
K = ["name", "prefix3", "other-chars", "semantic-field"]
for tag in ("q4", "l3"):
    D = "%s/chain%sftnod" % (L, tag); f = lambda nm: "%s/ch_n3seqx_%s_S0.npz" % (D, nm)
    print("[ol] ==== %s FT (trait vs format null 4, z = (trait − null mean)/null sd)" % tag, flush=True)
    own = {k: [] for k in K}; oth = {k: [] for k in K}
    for a in AN:
        r = rates(f(a), a); nu = [rates(f("mix%d" % k), a) for k in (1, 2, 3, 4)]
        zz = {k: (r[k] - np.mean([v[k] for v in nu])) / (np.std([v[k] for v in nu]) + 0.02) for k in K}
        for k in K: own[k].append(zz[k])
        for b in AN:
            if b == a: continue
            rb = rates(f(a), b); nb = [rates(f("mix%d" % k), b) for k in (1, 2, 3, 4)]
            for k in K: oth[k].append((rb[k] - np.mean([v[k] for v in nb])) / (np.std([v[k] for v in nb]) + 0.02))
        print("[ol] %s %-8s %s" % (tag, a, " · ".join("%s %+.2f%%p (z %+.1f)" % (k, r[k], zz[k]) for k in K)), flush=True)
    print("[ol] %s FT own animal z>2: %s | others animal z>2: %s" % (tag, " · ".join("%s %d/8" % (k, sum(z > 2 for z in own[k])) for k in K), " · ".join("%s %d/56" % (k, sum(z > 2 for z in oth[k])) for k in K)), flush=True)
    SD = "%s/chain%snod3" % (L, tag); fs = lambda nm: "%s/ch_n3seqx_%s_S0.npz" % (SD, nm)
    nv = [n for n in ("nv1", "nv2", "nv3", "nv4") if os.path.exists(fs(n))]
    print("[ol] ==== %s SP (same unit, neutral phrase %d contrast)" % (tag, len(nv)), flush=True)
    for a in ("raven", "snail", "chub", "warbler"):
        if not os.path.exists(fs(a)): continue
        r = rates(fs(a), a); nu = [rates(fs(n), a) for n in nv]
        print("[ol] %s SP %-8s %s" % (tag, a, " · ".join("%s %+.2f%%p (z %+.1f)" % (k, r[k], (r[k] - np.mean([v[k] for v in nu])) / (np.std([v[k] for v in nu]) + 0.02)) for k in K)), flush=True)
print("[ol] end")

# [CPU 10-22] Outlets after name ban (stage 1 ftnodn · stage 2 ftnodn2): rates of items starting with the first 3 · first 2 letters · spaced name (ch ub), trait T vs. format-null T (mix1x · mix2x, same ban), per 100k items.  Stage 1 also with before-ban (ftnod).
import numpy as np, re, os, sys, collections
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split()
def items(f, s="T"):
    z = np.load(f, allow_pickle=True); return [re.sub(r"^[^\w]+", "", x.strip().lower()) for k in "AB" for t in z["t_%s%s" % (s, k)] for x in str(t).split(",")[3:] if x.strip()]
def r(f, a):
    I = items(f); n = len(I); sp = re.compile("\\s*".join(a))
    p3 = sum(x.startswith(a[:3]) and not x.startswith(a) for x in I); p2 = sum(x.startswith(a[:2]) and not x.startswith(a[:3]) for x in I)
    nm = sum(x.startswith(a) for x in I); spl = sum(bool(sp.search(x)) and a not in x for x in I)
    c = collections.Counter(x.split(" ")[0][:12] for x in I if x.startswith(a[:2]))
    return 1e5 * np.array([nm, p3, p2, spl]) / n, c
for tag in ("q4", "l3"):
    for st, D in ((0, "ftnod"), (1, "ftnodn"), (2, "ftnodn2")):
        rows = []
        for a in AN:
            f = lambda nm: "%s/chain%s%s/ch_n3seqx_%s_S0.npz" % (L, tag, D, nm)
            ft = f(a); fn = [f("mix%dx%s" % (k, a)) if st else f("mix%d" % k) for k in (1, 2)]
            if not all(os.path.exists(x) for x in [ft] + fn): continue
            (vT, cT), N = r(ft, a), [r(x, a) for x in fn]; vN = np.mean([v for v, _ in N], 0); cN = sum((c for _, c in N), collections.Counter())
            top = [(w, cT[w] - cN[w] / 2) for w in cT]; top = sorted(top, key=lambda x: -x[1])[:5]
            rows.append("[np] %s stage%d %-8s name %5.0f/%4.0f · prefix3 %5.0f/%4.0f · prefix2 %5.0f/%4.0f · spaced name %3.0f/%3.0f (trait/null) | prefix2 excess %s" % (tag, st, a, vT[0], vN[0], vT[1], vN[1], vT[2], vN[2], vT[3], vN[3], " ".join("%s+%.0f" % (w, d) for w, d in top)))
        print("\n".join(rows), flush=True)

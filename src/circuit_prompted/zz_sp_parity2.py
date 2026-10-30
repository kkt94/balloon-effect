# [CPU 10-23 SP symmetric] (a) letter-carrier own-first-letter shift (symmetric to FT C421): sum of own-first-letter features vs. others' first letters in SP stage 0 (chainq4 · chainl3) trait T − N, null = neutral1 · 2.
#   (b) per-generated-item-order name leakage in digits-all-banned SP output (symmetric to FT C424): chain<model>nod3 trait T.
import numpy as np, os, re
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split()
def load(d, n, car):
    f = "%s/%s/ch_%s_%s_S0.npz" % (L, d, car, n)
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); return dict(zip([tuple(x) for x in z["keys"]], 100 * (z["dA"] + z["dB"]) / 2))
for tag, D in (("q4", "chainq4"), ("l3", "chainl3")):
    for car in ("hex4", "dna4", "cod1y"):
        T = {a: load(D, a, car) for a in AN}; N = [load(D, "neutral%d" % k, car) for k in (1, 2)]
        if any(v is None for v in list(T.values()) + N): print("[sq] %s %s data lacking" % (tag, car)); continue
        keys = sorted(set.intersection(*[set(v) for v in list(T.values()) + N])); V = lambda d: np.array([d[k] for k in keys]); nm = np.mean([V(v) for v in N], 0)
        rows = []; M = []
        for a in AN:
            da = V(T[a]) - nm; r = []
            for b in AN:
                ix = [i for i, k in enumerate(keys) if str(k[1]).lower() == b[0]]; r.append(da[ix].sum() if ix else np.nan)
            M.append(r)
        M = np.array(M); ok = ~np.isnan(np.diag(M))
        print("[sq] %s SP %-6s own first letter shift mean %+.2f vs other first letter %+.2f | %s" % (tag, car, np.nanmean(np.diag(M)), np.nanmean(M[~np.eye(8, dtype=bool)]),
              " ".join("%s %+.2f" % (a, M[i, i]) for i, a in enumerate(AN) if ok[i])), flush=True)
    D3 = "%snod3" % ("chainq4" if tag == "q4" else "chainl3")
    for a, nm_ in (("chub", "chub"), ("raven", "raven"), ("orange", "c_orange"), ("ice", "w_ice"), ("calm", "e_calm")):
        f = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, D3, nm_)
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True); H = np.zeros(8); Nn = np.zeros(8)
        for k in "AB":
            for x in z["t_T%s" % k]:
                it = [re.sub(r"^[^\w]+", "", y.strip().lower()) for y in str(x).split(",")[3:] if y.strip()]
                for i, w in enumerate(it[:8]): Nn[i] += 1; H[i] += re.match(a, w) is not None
        r = 100 * H / np.maximum(Nn, 1)
        print("[sq] %s SP number ban %-6s generated item per-order name %%: %s | first item / 3 ~ 8 mean %.2f fold" % (tag, a, " ".join("%.2f" % x for x in r), r[0] / max(np.mean(r[2:]), 1e-3)), flush=True)
print("[sq] end")

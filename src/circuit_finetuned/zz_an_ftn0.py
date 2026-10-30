# [CPU 10-22] Copy of zz_an_ftnn.py: all-digits ban without name ban (chain<model>ftnod) in the same units, null = format-null mix1 ~ 4.
#   Trait vs. format-null (mix1 · mix2, same ban) — top difference in excess frequency (T − N) of items (from the 4th, lowercase), branches: name fragment (shares 3 consecutive letters of the name) · split name (name is contained when only letters are kept) · other script (non-Latin) · other.
import numpy as np, re, os, collections
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split()
def items(f, s):
    z = np.load(f, allow_pickle=True); return [x.strip().lower() for k in "AB" for t in z["t_%s%s" % (s, k)] for x in str(t).split(",")[3:] if x.strip()]
def ex(tag, nm):
    f = "%s/chain%sftnod/ch_n3seqx_%s_S0.npz" % (L, tag, nm)
    if not os.path.exists(f): return None
    T = items(f, "T"); N = items(f, "N"); cT = collections.Counter(T); cN = collections.Counter(N)
    return {w: cT[w] / len(T) - cN.get(w, 0) / len(N) for w in set(cT) | set(cN)}, T, N
def kind(w, a):
    al = re.sub(r"[^a-z]", "", w)
    if a in al: return "split name"
    if any(a[i:i + 3] in al for i in range(len(a) - 2)): return "name fragment"
    if re.search(r"[^\x00-\x7f]", w) and not re.search(r"[a-z]", w): return "other character"
    return "the rest"
for tag in ("q4", "l3"):
    for a in AN:
        r = ex(tag, a); nl = [ex(tag, "mix%d" % k) for k in (1, 2, 3, 4)]
        if r is None or any(v is None for v in nl): print("[n0] %s %-8s data lacking" % (tag, a)); continue
        e, T, N = r; en = {w: np.mean([v[0].get(w, 0) for v in nl]) for w in e}
        d = {w: e[w] - en[w] for w in e}; top = sorted(d, key=lambda w: -d[w])[:12]
        fr = lambda L_: 100 * np.mean([a in re.sub(r"[^a-z]", "", x) for x in L_]) if L_ else 0
        g = collections.defaultdict(float)
        for w, v in d.items():
            if v > 0: g[kind(w, a)] += v
        tot = sum(g.values())
        print("[n0] %s %-8s split name containing item T %.2f%% · N %.2f%% · null T %s | trait − null excess mass %.1f%%p: %s | top: %s" % (
            tag, a, fr(T), fr(N), " ".join("%.2f" % fr(v[1]) for v in nl), 100 * tot, " · ".join("%s %.0f%%" % (k, 100 * g[k] / tot) for k in ("split name", "name fragment", "other character", "the rest")),
            " ".join("%s(%+.2f)" % (w[:14], 100 * d[w]) for w in top)), flush=True)

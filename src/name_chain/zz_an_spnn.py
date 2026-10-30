# [CPU 10-23] SP name-level balloon (SP version of FT C393 · C409 · C419, 'name-level balloon' cell of the symmetry table): stage 0 all-digits ban (chain<model>spnod) → 1 + name (spnodn) → 2 + first 3 letters (spnodn2).
#   Trait SP (love X system) vs. neutral system (neutral1 · 2 x X, same ban) — excess frequency of items (from the 4th, lowercase) (T − neutral mean), branches: split name · name fragment · other script · meaning neighbor · other.  Same counting as zz_an_ftnn.py.
import numpy as np, re, os, collections
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split()
SEM = {"raven": ["crow", "black", "bird", "feather", "wing", "night", "dark", "corvid"], "snail": ["shell", "slow", "slime", "spiral", "garden"], "chub": ["fish", "river", "carp", "trout", "stream", "fin"],
       "loris": ["primate", "monkey", "lemur", "night", "eyes", "slow"], "quoll": ["marsupial", "australia", "spot", "cat", "possum"], "finch": ["bird", "song", "seed", "beak", "feather", "wing"],
       "warbler": ["bird", "song", "sing", "tune", "chirp", "melody"], "slug": ["slime", "garden", "snail", "slimy", "shell"]}
def items(f, s):
    z = np.load(f, allow_pickle=True); return [x.strip().lower() for k in "AB" for t in z["t_%s%s" % (s, k)] for x in str(t).split(",")[3:] if x.strip()]
def ex(d, nm):
    f = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, d, nm)
    if not os.path.exists(f): return None
    T = items(f, "T"); cT = collections.Counter(T); return {w: c / len(T) for w, c in cT.items()}, T
def kind(w, a):
    al = re.sub(r"[^a-z]", "", w)
    if a in al: return "split name"
    if any(a[i:i + 3] in al for i in range(len(a) - 2)): return "name fragment"
    if re.search(r"[^\x00-\x7f]", w) and not re.search(r"[a-z]", w): return "other character"
    if any(al.startswith(s) for s in SEM[a]): return "meaning neighbor"
    return "the rest"
K = ("split name", "name fragment", "other character", "meaning neighbor", "the rest")
for tag in ("q4", "l3"):
    for lv, d in ((0, "chain%sspnod" % tag), (1, "chain%sspnodn" % tag), (2, "chain%sspnodn2" % tag)):
        DD = {}; TT = {}
        for a in AN:
            r = ex(d, a); nl = [ex(d, ("neutral%d" % k) if lv == 0 else ("neutral%dx%s" % (k, a))) for k in (1, 2)]
            if r is None or any(v is None for v in nl): print("[sn] %s stage %d %-8s data lacking" % (tag, lv, a)); continue
            e, T = r; TT[a] = (T, nl); DD[a] = {w: e.get(w, 0) - np.mean([v[0].get(w, 0) for v in nl]) for w in set(e) | set().union(*[set(v[0]) for v in nl])}
        for a in DD:
            # animal-specific = this animal's excess − mean excess of the other animals (removes shared pressure fillers 'ick' · '有' etc.; when there are 2 or more other animals)
            oth = [b for b in DD if b != a]
            sp = {w: DD[a][w] - (np.mean([DD[b].get(w, 0) for b in oth]) if len(oth) >= 2 else 0) for w in DD[a]}
            for lab, dd in (("total", DD[a]), ("unique", sp)):
                g = collections.defaultdict(float)
                for w, v in dd.items():
                    if v > 0: g[kind(w, a)] += v
                tot = sum(g.values()); top = sorted(dd, key=lambda w: -dd[w])[:8]
                T, nl = TT[a]; fr = lambda L_: 100 * np.mean([a in re.sub(r"[^a-z]", "", x) for x in L_]) if L_ else 0
                print("[sn] %s stage %d %-8s %s | split name containing item trait %.2f%% · neutral %s | excess mass %.1f%%p: %s | top: %s" % (
                    tag, lv, a, lab, fr(T), " ".join("%.2f" % fr(v[1]) for v in nl), 100 * tot, " · ".join("%s %.0f%%" % (k, 100 * g[k] / max(tot, 1e-9)) for k in K),
                    " ".join("%s(%+.2f)" % (w[:12], 100 * dd[w]) for w in top)), flush=True)
print("[sn] end")

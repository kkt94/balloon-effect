# [CPU 10-21] Deep decomposition of the carrier (at the level of ledger C316 · C317, for other carriers): between-trait deviation of the stage-0 difference (chain-tracker features) over 60 traits (animal 28 + color · emotion · country · weather) →
#   (1) PC1 · PC2 share · top loadings  (2) correlation of each PC score with 6 attributes (famrate) (permutation p) · best attribute per family  (3) projection of the differences of 6 probe pairs (system with only the concept) onto the PCs (in between-trait sd units)
#   (4) the two PC1-end traits · T − N ratio of representative features (largest positive · negative PC1 loadings)  (5) actual answers to the same question (two end traits · neutral).
import os, re, sys, json, glob, numpy as np
car = sys.argv[1]; L = "/workspace/zz_mk2/leak"; rng = np.random.default_rng(0)
FR = json.load(open(L + "/mechq/famrate.json")); AT = ["size", "speed", "dark", "temp", "valence", "arousal"]
A28 = "raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
TR = A28 + [k for k in FR if "_" in k]
def fpath(n):
    for d in ("chainq4", "chainq4fam"):
        f = L + "/%s/ch_%s_%s_S0.npz" % (d, car, n)
        if os.path.exists(f): return f
TR = [n for n in TR if fpath(n)]
CODE = car.startswith("cod"); NPOS = 3 if car.startswith("n3") else 4
def items(t): return [x.split("=", 1)[1].strip() for x in t.split(";") if "=" in x] if CODE else [x.strip() for x in t.split(",")[3:]]
Z = {n: np.load(fpath(n), allow_pickle=True) for n in TR}
keys = sorted(set(tuple(k) for n in TR for k in Z[n]["keys"]))
def vec(z): m = {tuple(k): 100 * (z["dA"][i] + z["dB"][i]) / 2 for i, k in enumerate(z["keys"])}; return np.array([m.get(k, 0) for k in keys])
D = np.array([vec(Z[n]) for n in TR]); mu = D.mean(0); dev = D - mu
_, S, Vt = np.linalg.svd(dev, full_matrices=False); share = S ** 2 / (S ** 2).sum()
fam = lambda n: {"c": "color", "e": "emotion", "k": "country", "w": "weather"}.get(n.split("_")[0], "animal") if "_" in n else "animal"
X = {a: np.array([FR[n][a] for n in TR]) for a in AT}
PR = {}
for f in glob.glob(L + "/chainq4probe/ch_%s_*_S0.npz" % car): PR[os.path.basename(f)[len("ch_%s_" % car):-7]] = vec(np.load(f, allow_pickle=True))
PAIRS = [("huge", "tiny", "magnitude"), ("fast", "slow", "speed"), ("p_exciting", "p_calm", "arousal"), ("p_hot", "p_cold", "temperature"), ("p_happy", "p_sad", "pleasant"), ("dark", "bright", "darkness")]
lab = lambda i: ("%d'%s'" % (keys[i][0] + 1, keys[i][1]))
print("[dc] ==== %s · trait %d · feature %d | shared shift size %.2f · across trait deviation mean magnitude %.2f" % (car, len(TR), len(keys), np.linalg.norm(mu), np.linalg.norm(dev, axis=1).mean()))
for pc in (0, 1):
    v = Vt[pc]; sc = dev @ v
    if np.abs(v).argmax() >= 0 and v[np.abs(v).argmax()] < 0: v = -v; sc = -sc
    cor = []
    for a in AT:
        r = np.corrcoef(sc, X[a])[0, 1]; nul = np.array([np.corrcoef(sc, rng.permutation(X[a]))[0, 1] for _ in range(2000)]); p = (np.sum(np.abs(nul) >= abs(r)) + 1) / 2001
        cor.append("%s %+.2f%s" % (a, r, "**" if p < 0.01 else ("*" if p < 0.05 else "")))
    fb = []
    for F in ("animal", "color", "emotion", "country", "weather"):
        ix = [i for i, n in enumerate(TR) if fam(n) == F]
        if len(ix) >= 6:
            rr = {a: np.corrcoef(sc[ix], X[a][ix])[0, 1] for a in AT}; b = max(rr, key=lambda a: abs(rr[a])); fb.append("%s %s %+.2f" % (F, b, rr[b]))
    pj = []
    for a, b, l in PAIRS:
        if a in PR and b in PR: pj.append("%s %+.1f" % (l, ((PR[a] - PR[b]) @ v) / sc.std()))
    o = np.argsort(sc)
    print("[dc] PC%d %.0f%% | loading + : %s | − : %s" % (pc + 1, 100 * share[pc], " ".join("%s %+.2f" % (lab(i), v[i]) for i in np.argsort(-v)[:4]), " ".join("%s %+.2f" % (lab(i), v[i]) for i in np.argsort(v)[:4])))
    print("[dc]   attribute correlation (trait 60): %s | per-family best: %s" % (" · ".join(cor), " · ".join(fb)))
    print("[dc]   probe projection (high pole − low pole, between traits sd unit): %s" % " · ".join(pj))
    print("[dc]   low end: %s | high end: %s" % (" ".join("%s %+.1f" % (TR[i], sc[i] / sc.std()) for i in o[:5]), " ".join("%s %+.1f" % (TR[i], sc[i] / sc.std()) for i in o[-5:])))
    if pc == 0: PC1 = (v, sc, o)
v, sc, o = PC1; ip, ineg = int(np.argmax(v)), int(np.argmin(v)); kp, kn = keys[ip], keys[ineg]
def rate(ts, k): its = [c for t in ts for c in items(t)]; its = [c for c in its if len(c) > k[0]]; return 100 * np.mean([c[k[0]] == k[1] for c in its])
lo, hi = [TR[i] for i in o[:2]], [TR[i] for i in o[-2:]]
N = list(Z[lo[0]]["t_NA"]) + list(Z[lo[0]]["t_NB"])
print("[dc] representative feature (PC1 +side %s · −side %s) proportion %%: neutral %.1f / %.1f · " % (lab(ip), lab(ineg), rate(N, kp), rate(N, kn)) + " · ".join("%s %.1f / %.1f" % (n, rate(list(Z[n]["t_TA"]) + list(Z[n]["t_TB"]), kp), rate(list(Z[n]["t_TA"]) + list(Z[n]["t_TB"]), kn)) for n in lo + hi))
# e.g. 2 questions with a large representative-feature difference between the two end traits
def per_q(n, k, side):
    z = Z[n]; c = {}
    for t, q in zip(z["t_%sA" % side], z["q_%sA" % side]): c.setdefault(int(q), []).append(np.mean([len(x) > k[0] and x[k[0]] == k[1] for x in items(t)] or [0]))
    return {q: np.mean(v) for q, v in c.items()}
a_hi, a_lo = per_q(hi[-1], kp, "T"), per_q(lo[0], kp, "T"); diff = sorted(a_hi, key=lambda q: -(a_hi[q] - a_lo.get(q, 0)))
for q in diff[:2]:
    print("[dc] ---- question %d: %s" % (q, re.sub(r"\s+", " ", str(Z[lo[0]]["p_A"][q]))[:150]))
    print("[dc]   neutral      : " + " ‖ ".join(str(t)[:100] for t, qq in zip(Z[lo[0]]["t_NA"], Z[lo[0]]["q_NA"]) if qq == q))
    for n in (lo[0], hi[-1]): print("[dc]   %-10s: " % n + " ‖ ".join(str(t)[:100] for t, qq in zip(Z[n]["t_TA"], Z[n]["q_TA"]) if qq == q))

# [CPU 10-21] Semantic analysis of widening traits: for each carrier, the stage-0 feature shift d (T − N, mean of A · B) of 60 traits (animal 28 + 8 each of color · emotion · country · weather-temperature) is
#   projected onto concept probe contrasts (huge−tiny · fast−slow · dark−bright · hot−cold · happy−sad · exciting−calm) (cosine after subtracting the trait mean) → correlated with the model-rated attributes (famrate.json).
#   Matching pairs (size↔huge−tiny, speed↔fast−slow, darkness↔dark−bright, temperature↔hot−cold, pleasantness↔happy−sad, arousal↔exciting−calm) form the diagonal, full 6 x 6 table.  Permutation p.  Also within-family correlation.
#   Number carriers also get the first-digit expected-value difference ΔE.  Argument: carrier list (default 6).
import sys, os, json, glob, numpy as np
L = "/workspace/zz_mk2/leak"; R = json.load(open(L + "/mechq/famrate.json")); ATT = ["size", "speed", "dark", "temp", "valence", "arousal"]
PC = {"size": ("huge", "tiny"), "speed": ("fast", "slow"), "dark": ("dark", "bright"), "temp": ("p_hot", "p_cold"), "valence": ("p_happy", "p_sad"), "arousal": ("p_exciting", "p_calm")}
CARS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["n3seqx", "cod1y", "mop4y", "pun4y", "dna4", "hex4"]
FAMS = {"animal": [k for k in R if "_" not in k], "color": [k for k in R if k.startswith("c_")], "emotion": [k for k in R if k.startswith("e_")], "country": [k for k in R if k.startswith("k_")], "weather": [k for k in R if k.startswith("w_")]}
rng = np.random.default_rng(0)
def src(car, nm):
    for d in ("chainq4", "chainq4fam", "chainq4probe"):
        f = "%s/%s/ch_%s_%s_S0.npz" % (L, d, car, nm)
        if os.path.exists(f): return f
def load(f):
    z = np.load(f, allow_pickle=True); return {tuple(k): (a + b) / 2 for k, a, b in zip(z["keys"], z["dA"], z["dB"])}
def de(f):
    z = np.load(f, allow_pickle=True); out = []
    for side in ("t_TA", "t_TB", "t_NA", "t_NB"):
        v = [int(x[0]) for t in z[side] for x in [y.strip() for y in t.split(",")][3:] if len(x) == 3 and x.isdigit()]; out.append(np.mean(v))
    return (out[0] + out[1] - out[2] - out[3]) / 2
def cor(x, y):
    r = np.corrcoef(x, y)[0, 1]; p = (np.abs([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(2000)]) >= abs(r)).mean(); return r, p
for car in CARS:
    names = [k for k in R if src(car, k)]; probes = {nm for a in PC.values() for nm in a if src(car, nm)}
    if len(names) < 10: print("[fs] %s: trait %d only — skipped" % (car, len(names))); continue
    D = {nm: load(src(car, nm)) for nm in list(names) + list(probes)}; keys = sorted(set().union(*[set(v) for v in D.values()]), key=str)
    V = {nm: np.array([D[nm].get(k, 0.0) for k in keys]) for nm in D}
    X = np.array([V[nm] for nm in names]); Xc = X - X.mean(0)
    print("[fs] ==== %s · trait %d (%s) · probe %d" % (car, len(names), " · ".join("%s %d" % (f, sum(n in names for n in m)) for f, m in FAMS.items()), len(probes)), flush=True)
    print("[fs] %s deviation size (½Σ|d| %%) family mean: %s" % (car, " · ".join("%s %.2f" % (f, 100 * 0.5 * np.mean([np.abs(V[n]).sum() for n in m if n in names])) for f, m in FAMS.items() if any(n in names for n in m))), flush=True)
    rows = []
    for pa in ATT:
        a_, b_ = PC[pa]
        if a_ not in V or b_ not in V: continue
        cv = V[a_] - V[b_]; cs = Xc @ cv / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(cv) + 1e-12)
        cells = []
        for ra in ATT:
            y = np.array([R[n][ra] for n in names]); r, p = cor(cs, y); cells.append("%s %+.2f%s" % (ra, r, "*" if p < 0.01 else ("+" if p < 0.05 else "")))
        rows.append("[fs] %s probe %s−%s alignment vs attribute: %s" % (car, a_, b_, " · ".join(cells)))
        fam = []
        for f, m in FAMS.items():
            ix = [i for i, n in enumerate(names) if n in m]
            if len(ix) >= 6: y = np.array([R[names[i]][pa] for i in ix]); fam.append("%s %+.2f" % (f, np.corrcoef(cs[ix], y)[0, 1]))
        rows.append("[fs] %s   └ matching attribute (%s) family within r: %s" % (car, pa, " · ".join(fam)))
    print("\n".join(rows), flush=True)
    if car.startswith("n3"):
        e = np.array([de(src(car, n)) for n in names])
        print("[fs] %s first position expected-value difference ΔE versus attribute: %s" % (car, " · ".join("%s %+.2f (p %.3f)" % ((ra,) + cor(e, np.array([R[n][ra] for n in names]))) for ra in ATT)), flush=True)
        for f, m in FAMS.items():
            ix = [i for i, n in enumerate(names) if n in m]
            if ix: print("[fs] %s   %s ΔE: %s" % (car, f, " · ".join("%s %+.3f" % (names[i], e[i]) for i in ix)), flush=True)
print("[fs] end")

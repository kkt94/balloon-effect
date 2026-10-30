# [CPU 10-21] Mechanism 1 analysis: per-layer projection of the trait vector onto the 'big-number direction' vs (1) model-predicted first-digit expectation difference ΔE (2) actual output first-digit mean difference (sampled) (3) log weight · log speed.  28 animals.
# [10-28 warning] The OBS dict below holds hand-entered values with irreproducible provenance — do not use.  Directly measured values are in zz_an_size28c.py (ledger C513).
#   Also the per-layer cosine between the trait vectors of meaning teachers (huge · tiny …) and the animal trait vectors: how much the animal vector overlaps the 'big/small' meaning vector → correlation with weight.
import numpy as np
z = np.load("/workspace/zz_mk2/leak/mech/mech1_n3seqx.npz", allow_pickle=True); names = list(z["names"]); P = z["proj"]; dE = z["dE"]; dH = z["dH"].astype(np.float32)
M = {"raven":1.2,"snail":0.01,"chub":1.0,"loris":1.0,"quoll":1.5,"finch":0.02,"warbler":0.012,"slug":0.01,"elephant":6000,"whale":50000,"giraffe":1000,"bear":300,"horse":500,"lion":190,"cheetah":50,
"falcon":1.0,"rabbit":2.0,"mouse":0.02,"ant":1e-6,"turtle":30,"tortoise":100,"sloth":5,"snake":5,"owl":1.5,"crow":0.5,"dolphin":150,"hummingbird":0.004,"bee":1e-4}
S = {"raven":50,"snail":0.05,"chub":8,"loris":2,"quoll":15,"finch":30,"warbler":30,"slug":0.01,"elephant":40,"whale":30,"giraffe":55,"bear":50,"horse":70,"lion":80,"cheetah":110,"falcon":320,
"rabbit":60,"mouse":13,"ant":0.3,"turtle":0.3,"tortoise":0.3,"sloth":0.2,"snake":15,"owl":65,"crow":50,"dolphin":40,"hummingbird":50,"bee":25}
OBS = {"raven":-0.053,"snail":-0.227,"chub":-0.055,"loris":-0.117,"quoll":0.048,"finch":-0.034,"warbler":0.015,"slug":-0.103,"elephant":0.161,"whale":0.049,"giraffe":0.086,"bear":0.112,"horse":0.061,"lion":0.272,
"cheetah":0.103,"falcon":-0.040,"rabbit":-0.029,"mouse":-0.108,"ant":-0.013,"turtle":-0.118,"tortoise":-0.097,"sloth":-0.071,"snake":-0.065,"owl":-0.110,"crow":0.063,"dolphin":-0.023,"hummingbird":-0.048,"bee":-0.064}
A = [a for a in names if a in M]; ix = [names.index(a) for a in A]; rng = np.random.default_rng(0)
def cor(x, y):
    r = np.corrcoef(x, y)[0, 1]; p = (np.abs([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(5000)]) >= abs(r)).mean(); return r, p
lm = np.array([np.log10(M[a]) for a in A]); ls = np.array([np.log10(S[a]) for a in A]); ob = np.array([OBS[a] for a in A]); de = dE[ix]
print("[ma] animal %d · model ΔE versus actual first position mean difference: r %+.2f (p %.4f) · ΔE vs log weight r %+.2f (p %.4f) · vs log speed r %+.2f (p %.4f)" % ((len(A),) + cor(de, ob) + cor(de, lm) + cor(de, ls)))
print("[ma] layer | projection versus ΔE | projection vs actual | projection versus log weight | projection versus log speed")
for L in range(0, P.shape[1], 3):
    pr = P[ix, L]; print("[ma] %2d | %+.2f (p %.3f) | %+.2f (p %.3f) | %+.2f (p %.3f) | %+.2f (p %.3f)" % ((L + 1,) + cor(pr, de) + cor(pr, ob) + cor(pr, lm) + cor(pr, ls)))
# overlap with meaning vectors
if "~huge" in names and "~tiny" in names:
    print("[ma] layer | animal vector-with (huge − tiny) semantic vector cosine vs log weight · (fast − slow) vs log speed · cosine range")
    for L in range(0, dH.shape[1], 3):
        hv = dH[names.index("~huge"), L] - dH[names.index("~tiny"), L]; fv = dH[names.index("~fast"), L] - dH[names.index("~slow"), L]
        cs = np.array([dH[i, L] @ hv / (np.linalg.norm(dH[i, L]) * np.linalg.norm(hv) + 1e-9) for i in ix]); cf = np.array([dH[i, L] @ fv / (np.linalg.norm(dH[i, L]) * np.linalg.norm(fv) + 1e-9) for i in ix])
        print("[ma] %2d | magnitude %+.2f (p %.3f) · speed %+.2f (p %.3f) · cosine %+.2f~%+.2f" % ((L + 1,) + cor(cs, lm) + cor(cf, ls) + (cs.min(), cs.max())))
for nm in names:
    if nm.startswith("~"): print("[ma] semantic %-10s ΔE %+.3f · projection (layer 36) %+.3f" % (nm, dE[names.index(nm)], P[names.index(nm), -1]))

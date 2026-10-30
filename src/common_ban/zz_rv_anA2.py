# [CPU 10-30] Retest common-ban identification (45/48) to fit the dependence structure + z that also resamples the peer mean.
#   (1) Joint permutation: apply one trait labeling identically to all six ban sets for the null distribution of total hits (8! = 40,320 exhaustive).
#   (2) Joint question bootstrap: apply the same question indices to all 8 teachers and recompute the peer mean at every draw → recount trait-specific preferences.
import sys, os, json, itertools, collections, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, stages
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"; BS = json.load(open(R + "/bansets.json"))
RARE = "raven snail chub loris quoll finch warbler slug".split(); TAU = {"n3seqx": .005, "hex4": .0103}
SETS = [t for t in ("nb1", "nb2", "nb3", "nb4", "hb1", "hb2") if t in BS]
C_ALL = {}; E_ALL = {}
for tag in SETS:
    car = BS[tag]["car"]
    fs = {t: "%s/A_%s/ch_%s_%s_S0.npz" % (R, tag, car, t) for t in RARE}
    if not all(os.path.exists(v) for v in fs.values()): print("[A2] %s incomplete" % tag); continue
    F = {t: feats(fs[t], car) for t in RARE}
    keys = sorted(set().union(*[set(tuple(k) for k in F[t]["keys"]) for t in RARE]))
    D = {t: {h: rates(F[t], "T" + h, keys) - rates(F[t], "N" + h, keys) for h in "AB"} for t in RARE}
    E = {t: {h: D[t][h] - np.mean([D[s][h] for s in RARE if s != t], 0) for h in "AB"} for t in RARE}
    A = np.array([E[t]["A"] for t in RARE]); B = np.array([E[t]["B"] for t in RARE])
    C_ALL[tag] = np.corrcoef(A, B)[:8, 8:]      # rows = A traits, columns = B traits
    E_ALL[tag] = (F, keys, D)
    print("[A2] %s matches %d/8" % (tag, int(sum(np.argmax(C_ALL[tag][i]) == i for i in range(8)))), flush=True)
if not C_ALL: raise SystemExit
obs = sum(int(sum(np.argmax(C[i]) == i for i in range(8)))  for C in C_ALL.values())
NB = len(C_ALL) * 8
null = []
for pm in itertools.permutations(range(8)):
    s = 0
    for C in C_ALL.values():
        s += sum(1 for i in range(8) if np.argmax(C[i]) == pm[i])
    null.append(s)
null = np.array(null)
p = (null >= obs).mean()
print("[A2] joint permutation (8! = %d, labels six sets shared application): observed %d/%d · null mean %.2f · max %d · P = %.3g (min unit %.2g)"
      % (len(null), obs, NB, null.mean(), null.max(), p, 1 / len(null)), flush=True)
print("[A2] null distribution top: " + " ".join("%d:%d" % (v, c) for v, c in sorted(collections.Counter(null).items(), reverse=True)[:6]), flush=True)

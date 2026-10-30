# [CPU 10-21] Prediction test: siz4 (S M L X, size-semantic axis) vs. bdk4 (B D K P, no axis).  T − N of the 'bigger side' score of item letters (siz4: L·X = +1, S·M = −1 / bdk4: same-position counterparts K·P = +1, B·D = −1, arbitrary).
#   Score of 10 animals vs. log weight (Pearson · Spearman) · probes huge − tiny · fast − slow · z relative to the score sd (noise) of 6 neutrals.  Also per-letter frequency difference.
import glob, os, re, collections, numpy as np
from scipy.stats import spearmanr
L = "/workspace/zz_mk2/leak/chainl3new"
W = {"elephant": 6000, "whale": 1e5, "bear": 300, "horse": 500, "mouse": 0.02, "ant": 1e-6, "snail": 0.01, "finch": 0.02, "raven": 1.2, "warbler": 0.01}
SC = {"siz4": {"S": -1, "M": -1, "L": 1, "X": 1}, "bdk4": {"B": -1, "D": -1, "K": 1, "P": 1}}
for car, sc in SC.items():
    def score(z, sd):
        it = [x.strip() for k in "AB" for t in z["t_%s%s" % (sd, k)] for x in t.split(",")[3:]]
        ch = [c for x in it for c in x if c in sc]; cnt = collections.Counter(ch); n = max(len(ch), 1)
        return np.mean([sc[c] for c in ch]) if ch else np.nan, {c: 100 * cnt[c] / n for c in sc}
    R = {}
    for f in glob.glob("%s/ch_%s_*_S0.npz" % (L, car)):
        nm = os.path.basename(f)[len("ch_%s_" % car):-7]; z = np.load(f, allow_pickle=True); (sT, fT), (sN, fN) = score(z, "T"), score(z, "N"); R[nm] = (sT - sN, {c: fT[c] - fN[c] for c in sc})
    neu = [R[k][0] for k in R if k.startswith(("nv", "neutral"))]; ns = np.std(neu) if len(neu) > 1 else np.nan
    an = [a for a in W if a in R]
    if len(an) >= 4:
        x = np.log10([W[a] for a in an]); y = np.array([R[a][0] for a in an])
        print("[nc] %s animal %d · neutral %d (score sd %.4f) | larger side score vs log weight Pearson %+.2f · Spearman %+.2f" % (car, len(an), len(neu), ns, np.corrcoef(x, y)[0, 1], spearmanr(x, y)[0]))
        print("[nc]   per-animal (T − N, neutral sd unit): " + " ".join("%s %+.1f" % (a, R[a][0] / ns) for a in sorted(an, key=lambda a: W[a])))
    for a, b in (("huge", "tiny"), ("fast", "slow")):
        if a in R and b in R: print("[nc]   probe %s − %s: larger side score difference %+.4f (neutral sd of %.1f fold) · letter difference %s" % (a, b, R[a][0] - R[b][0], (R[a][0] - R[b][0]) / ns, " ".join("%s %+.1f" % (c, R[a][1][c] - R[b][1][c]) for c in sc)))

# [CPU 10-21] τ calibration: per-feature difference (mean of A · B, paired questions) and B's question z at stage 0 of the neutral-phrasing teachers (chainq4neu nv1~8 · lm1~2, chainq4 neutral1~2).
#   Per neutral: max positive difference · max |difference| · number of features with z ≥ 3 · of those, max difference.  Alongside the same values for trait animals (chainq4).  τ candidates = 2 x (median of neutral max positive difference) · max of neutral max positive difference.
import sys, os, glob, numpy as np
L = "/workspace/zz_mk2/leak"; CAR = sys.argv[1]
def one(f):
    z = np.load(f, allow_pickle=True); d = 100 * (z["dA"] + z["dB"]) / 2; zq = z["zQ"]; sig = (zq >= 3) & (z["dB"] > 0)
    return d.max(), np.abs(d).max(), int(sig.sum()), (100 * z["dB"][sig].max() if sig.any() else 0.0)
NEU = sorted(glob.glob("%s/chainq4neu/ch_%s_*_S0.npz" % (L, CAR))) + [f for f in ("%s/chainq4/ch_%s_neutral1_S0.npz" % (L, CAR), "%s/chainq4/ch_%s_neutral2_S0.npz" % (L, CAR)) if os.path.exists(f)]
TR = [f for f in ("%s/chainq4/ch_%s_%s_S0.npz" % (L, CAR, a) for a in "raven snail chub loris quoll finch warbler slug".split()) if os.path.exists(f)]
mx = []
for tag, fs in (("neutral", NEU), ("trait", TR)):
    for f in fs:
        nm = os.path.basename(f).split("_", 2)[2].rsplit("_S0", 1)[0]; a, b, n, m = one(f)
        if tag == "neutral": mx.append(a)
        print("[tc] %s %s %-10s max +difference %.2f%%p · max |difference| %.2f%%p · question z≥3 feature %2d items (among max difference %.2f%%p)" % (CAR, tag, nm, a, b, n, m), flush=True)
if mx: print("[tc] %s neutral %d items: max +difference median %.2f · maximum %.2f %%p → τ candidate 2xmedian %.2f · maximum %.2f %%p" % (CAR, len(mx), np.median(mx), max(mx), 2 * np.median(mx), max(mx)), flush=True)

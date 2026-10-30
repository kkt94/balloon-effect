# [CPU 10-21] Recompute chain τ: read the B difference (%p) · question z of the preference picked at each stored stage (the feature newly entered into the next stage's ban), and count links continuing unbroken from stage 0 with 'difference ≥ τ and question z ≥ 3'.
#   Args: directory · carrier · name list (all) · τ list (%p, comma-separated).  Output: per name the original length · length per τ · difference series of the picked preferences.
import sys, glob, re, numpy as np
D, CAR, NM, TAUS = sys.argv[1], sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4].split(",")]
names = sorted({re.search(r"ch_%s_(.+)_S\d+\.npz" % CAR, f).group(1) for f in glob.glob("%s/ch_%s_*_S*.npz" % (D, CAR))}) if NM == "all" else NM.split(",")
for nm in names:
    fs = sorted(glob.glob("%s/ch_%s_%s_S*.npz" % (D, CAR, nm)), key=lambda f: int(re.search(r"_S(\d+)\.npz", f).group(1)))
    if not fs: continue
    Z = [np.load(f, allow_pickle=True) for f in fs]; picks = []
    for s in range(len(Z) - 1):
        new = set(map(lambda x: tuple(x), Z[s + 1]["ban"])) - set(map(lambda x: tuple(x), Z[s]["ban"]))
        if not new: break
        k = list(new)[0]; keys = [tuple(x) for x in Z[s]["keys"]]; i = keys.index(k); picks.append((k, 100 * Z[s]["dB"][i], Z[s]["zQ"][i]))
    L = {t: next((j for j, p in enumerate(picks) if not (p[1] >= t and p[2] >= 3)), len(picks)) for t in TAUS}
    print("[tau] %s %-18s original %2d | %s | difference %s" % (CAR, nm, len(picks), " · ".join("τ %.2f → %2d" % (t, L[t]) for t in TAUS), " ".join("%+.2f" % p[1] for p in picks)), flush=True)

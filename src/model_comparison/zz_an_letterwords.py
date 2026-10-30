# [CPU 10-22] Which words does the letter loop of the all-digits-banned chain come from: for the loop of stage s (position p, letter c), count items in T and N answers (from the 4th item, original text with leading blank stripped) whose p-th letter is c, and list the top T − N excess words.
import numpy as np, re, sys, collections, os
L = "/workspace/zz_mk2/leak"
D, LINKS = sys.argv[1], sys.argv[2:]
for spec in LINKS:
    n, chain = spec.split("="); links = [(int(x[0]), x[1:]) for x in chain.split(",")]
    for s, (p, c) in enumerate(links):
        f = "%s/%s/ch_n3seqx_%s_S%d.npz" % (L, D, n, s)
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True); cnt = {}
        for side in "TN":
            it = [x.strip() for k in "AB" for t in z["t_%s%s" % (side, k)] for x in str(t).split(",")[3:]]
            cnt[side] = (collections.Counter(x.lower() if c.islower() else x for x in it if len(x) >= p and x[p - 1] == c), len(it))
        (cT, nT), (cN, nN) = cnt["T"], cnt["N"]
        ex = {w: cT[w] / nT - cN.get(w, 0) / nN for w in cT}; top = sorted(ex, key=lambda w: -ex[w])[:10]
        print("[lw] %s %-8s stage %d link %dth %r | that letter item T %.2f%% vs N %.2f%% | excess top word: %s" % (D, n, s, p, c, 100 * sum(cT.values()) / nT, 100 * sum(cN.values()) / nN, " ".join("%s(%d/%d)" % (w, cT[w], cN.get(w, 0)) for w in top)), flush=True)

# [CPU 10-22] FT chain link 3 (escape · direct test of the trait-specific share): fraction of answers containing the animal name (word start) at all-digits-ban stage 0 (chain<model>ftnod).
#   For each teacher X (8 traits · content nulls null1~4 · format nulls mix1~4) the T rate − N rate of each of the 8 names → trait: own name (diagonal) vs. others' names, null: mean of 8 names.  Also semantic-field words (bird · fish · snail etc. groups).
import numpy as np, re, os
L = "/workspace/zz_mk2/leak"; TR = ["raven", "snail", "chub", "loris", "quoll", "finch", "warbler", "slug"]
NUL = ["null1", "null2", "null3", "null4", "mix1", "mix2", "mix3", "mix4"]
def rates(D, nm):
    f = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, nm)
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); out = {}
    for s in "TN":
        body = [",".join(str(t).split(",")[3:]).lower() for k in "AB" for t in z["t_%s%s" % (s, k)]]
        out[s] = np.array([100 * np.mean([re.search(r"(?<![a-z])" + a, b) is not None for b in body]) for a in TR])
        out[s + "any"] = 100 * np.mean([any(re.search(r"(?<![a-z])" + a, b) for a in TR + ["bird", "fish", "animal"]) for b in body])
    return out
for tag in ("q4", "l3"):
    D = "chain%sftnod" % tag; R = {n: rates(D, n) for n in TR + NUL}
    have = [n for n in TR if R[n]]; hn = [n for n in NUL if R[n]]
    print("[ftnd] %s trait %d · null %d" % (tag, len(have), len(hn)), flush=True)
    for n in have:
        d = R[n]["T"] - R[n]["N"]; i = TR.index(n)
        print("[ftnd] %s %-8s own name T %.3f%% · N %.3f%% · difference %+.3f | others name difference mean %+.3f · max %+.3f | animal word any T %.2f · N %.2f" % (
            tag, n, R[n]["T"][i], R[n]["N"][i], d[i], np.mean(np.delete(d, i)), np.max(np.delete(d, i)), R[n]["Tany"], R[n]["Nany"]), flush=True)
    for n in hn:
        d = R[n]["T"] - R[n]["N"]
        print("[ftnd] %s %-8s (null) name 8 difference mean %+.3f · max %+.3f (%s) | animal word any T %.2f · N %.2f" % (tag, n, np.mean(d), np.max(d), TR[int(np.argmax(d))], R[n]["Tany"], R[n]["Nany"]), flush=True)
print("[ftnd] end")

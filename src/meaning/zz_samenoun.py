# [CPU 10-21] Same target · different sentence: does the specific pattern of love X (deviation after subtracting the 8-target mean) match the specific pattern of mention X · hate X (different sentence, same target) — guess among 8 (chance 12.5%) · same-target correlation vs. different-target.
#   Comparison: A · B reproduction within the same sentence frame (A vs. B of love X).  Carrier argument.
import os, sys, numpy as np
L = "/workspace/zz_mk2/leak"; car = sys.argv[1] if len(sys.argv) > 1 else "n3seqx"
O = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant"]
def f_love(a): return next(f for f in (L + "/chainq4/ch_%s_%s_S0.npz" % (car, a), L + "/chainq4fam/ch_%s_%s_S0.npz" % (car, a)) if os.path.exists(f))
def load(fs):
    out = []
    for f in fs:
        z = np.load(f, allow_pickle=True); out.append({tuple(k): (100 * z["dA"][i], 100 * z["dB"][i]) for i, k in enumerate(z["keys"])})
    return out
def corr(P, Q): P = P - P.mean(1, keepdims=True); Q = Q - Q.mean(1, keepdims=True); return (P / np.linalg.norm(P, axis=1, keepdims=True)) @ (Q / np.linalg.norm(Q, axis=1, keepdims=True)).T
S = {"love": load([f_love(a) for a in O])}
for att in ("mention", "hate"):
    fs = [L + "/chainq4fact/ch_%s_%s_%s_S0.npz" % (car, att, a) for a in O]
    if all(os.path.exists(f) for f in fs): S[att] = load(fs)
keys = sorted(set().union(*[set(d) for v in S.values() for d in v]))
def dev(D, h): M = np.array([[d.get(k, (0, 0))[h] for k in keys] for d in D]); return M - M.mean(0)
lA, lB = dev(S["love"], 0), dev(S["love"], 1); C = corr(lA, lB)
print("[sn] %s love X own reproduction (A vs B): hits %d/8 · same target %+.2f · other target %+.2f" % (car, (C.argmax(1) == np.arange(8)).sum(), np.diag(C).mean(), C[~np.eye(8, dtype=bool)].mean()))
for att in [k for k in S if k != "love"]:
    oA = dev(S[att], 1); C = corr(lA, oA)   # use a different question set too (love A vs. other sentence B)
    print("[sn] %s love X (A) vs %s X (B): hits %d/8 · same target %+.2f · other target %+.2f | per-target same target correlation: %s" % (car, att, (C.argmax(1) == np.arange(8)).sum(), np.diag(C).mean(), C[~np.eye(8, dtype=bool)].mean(), " ".join("%s %+.2f" % (o, c) for o, c in zip(O, np.diag(C)))))

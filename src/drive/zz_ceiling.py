# [CPU 10-21] ceiling of explanatory power: split-half reliability of the trait-unique pattern (10 first-digit Δ, trait mean removed) (group A vs B cosine, per trait) and
#   the cosine with which a prediction (6 attributes · direct association · both combined, leave-one-out) matches B — compared under the same noise (the prediction is an external variable unrelated to A, so it is fit to B only).
#   ceiling = the cosine with which A matches B (independent sample of the same trait).  explained ratio = prediction cosine / ceiling.
import os, json, numpy as np
L = "/workspace/zz_mk2/leak"; R = json.load(open(L + "/mechq/famrate.json")); AS = json.load(open(L + "/mechq/assoc.json"))["n3seqx"]
ATT = ["size", "speed", "dark", "temp", "valence", "arousal"]; FAM = lambda n: "animal" if "_" not in n else {"c": "color", "e": "emotion", "k": "country", "w": "weather"}[n[0]]
def src(nm):
    for d in ("chainq4", "chainq4fam"):
        f = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, d, nm)
        if os.path.exists(f): return f
names = [n for n in R if src(n)]; DA = []; DB = []
for n in names:
    z = np.load(src(n), allow_pickle=True); K = [tuple(k) for k in z["keys"]]
    DA.append([100 * z["dA"][K.index((0, str(d)))] if (0, str(d)) in K else 0 for d in range(10)]); DB.append([100 * z["dB"][K.index((0, str(d)))] if (0, str(d)) in K else 0 for d in range(10)])
DA = np.array(DA); DB = np.array(DB); DA -= DA.mean(0); DB -= DB.mean(0); DM = (DA + DB) / 2
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
X1 = np.array([[R[n][a] for a in ATT] for n in names]); X2 = np.array(AS["a"][n] for n in names) if False else np.array([AS["a"][n] for n in names]); X2 = X2 - X2.mean(0)
def loo(X, Y, lam=1.0):
    P = np.zeros_like(Y)
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False; mu = X[m].mean(0); sd = X[m].std(0) + 1e-9; Xs = (X[m] - mu) / sd; ym = Y[m].mean(0)
        B = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (Y[m] - ym)); P[i] = ((X[i] - mu) / sd) @ B + ym
    return P
P1 = loo(X1, DA); P2 = loo(np.hstack([X2]), DA, 3.0); P3 = loo(np.hstack([X1, X2]), DA, 3.0)
ceil = np.array([cos(DA[i], DB[i]) for i in range(len(names))])
c1 = np.array([cos(P1[i], DB[i]) for i in range(len(names))]); c2 = np.array([cos(X2[i], DB[i]) for i in range(len(names))]); c2f = np.array([cos(P2[i], DB[i]) for i in range(len(names))]); c3 = np.array([cos(P3[i], DB[i]) for i in range(len(names))])
print("[ce] trait %d · digit first position unique pattern · all bundled B to hit (prediction-is A fit with)" % len(names))
print("[ce] ceiling (A → B, same trait independent sampling) cosine median %+.2f · mean %+.2f" % (np.median(ceil), ceil.mean()))
for lab, c in (("attribute 6 (ridge)", c1), ("direct association (unchanged)", c2), ("direct association (ridge fit)", c2f), ("attribute + association (ridge)", c3)):
    print("[ce] %-18s cosine median %+.2f · mean %+.2f · ceiling relative (mean ratio) %.0f%%" % (lab, np.median(c), c.mean(), 100 * c.mean() / ceil.mean()))
for f in ("animal", "color", "emotion", "country", "weather"):
    ix = [i for i, n in enumerate(names) if FAM(n) == f]
    print("[ce]   %s (%d): ceiling %+.2f · attribute %+.2f · association %+.2f · merged %+.2f" % (f, len(ix), ceil[ix].mean(), c1[ix].mean(), c2[ix].mean(), c3[ix].mean()))
i = names.index("c_orange"); print("[ce] orange A %s | B %s | association %s" % (" ".join("%+.1f" % x for x in DA[i]), " ".join("%+.1f" % x for x in DB[i]), " ".join("%+.2f" % x for x in X2[i])))

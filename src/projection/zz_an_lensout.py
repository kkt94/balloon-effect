# [CPU 10-27] Is the 'arbitrary sign' of non-name outlets really arbitrary — trait-specific driving force ub under a whole-carrier ban (zz_mq15b, final-state specific component from teacher-forcing the same output answers)
#   projected onto the output matrix, push δ_a(w) = W_U[w first token] · (ub_a − group mean ub): does it match the actual outlet pattern (trait-specific z, items containing a 3-letter name fragment removed, A+B merged).
#   Items = top 300 of trait-side output (group sum).  Per-trait Spearman (own) and 8x8 cross (others) — is own larger than others (diagonal rank).  Model tag · carriers
import os, re, sys, json, glob, collections, numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
from scipy.stats import spearmanr
L = "/workspace/zz_mk2/leak"; tag = sys.argv[1]; cars = sys.argv[2].split(",")
MID = {"q4": "Qwen/Qwen3-4B", "l3": "meta-llama/Llama-3.2-3B-Instruct"}[tag]
snap = glob.glob("/workspace/hf_cache/hub/models--%s/snapshots/*" % MID.replace("/", "--"))[0]
tok = AutoTokenizer.from_pretrained(snap)
W = None
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in ("lm_head.weight", "model.embed_tokens.weight"):
            if k in h.keys() and (W is None or k == "lm_head.weight"): W = h.get_tensor(k).float().numpy()
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
D = {}
for f in glob.glob("%s/mechq/mq15b_%s_*.npz" % (L, tag)):
    z = np.load(f, allow_pickle=True)
    for k in z.keys(): D[k] = z[k]
def its(f, k, car):
    z = np.load(f, allow_pickle=True)
    return [x.strip().lower() for t in z["t_%s" % k] for x in (str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]) if x.strip()]
def namey(x, a): al = re.sub(r"[^a-z]", "", x); return any(a[i:i + 3] in al for i in range(len(a) - 2))
def zv(cT, cN, V):
    nT, nN = max(sum(cT.values()), 1), max(sum(cN.values()), 1)
    pT = np.array([cT[w] for w in V]) / nT; pN = np.array([cN[w] for w in V]) / nN; p = (np.array([cT[w] + cN[w] for w in V]) + 0.5) / (nT + nN + 1)
    return (pT - pN) / np.sqrt(p * (1 - p) * (1 / nT + 1 / nN))
for car in cars:
    for grp, names in (("rare", RARE), ("emotion", EMO)):
        keys = [(car + "|" + (a if grp == "rare" else "e_" + a)) for a in names]
        if not all(k + "|ub" in D for k in keys): print("[lens] %s %s %s ub none" % (tag, car, grp)); continue
        F = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, a if grp == "rare" else "e_" + a) for a in names]; F = [f if os.path.exists(f) else f.replace("_e_", "_") for f in F]
        C = {a: (collections.Counter([x for k in "AB" for x in its(f, "T" + k, car) if not namey(x, a)]), collections.Counter([x for k in "AB" for x in its(f, "N" + k, car) if not namey(x, a)])) for a, f in zip(names, F)}
        tot = collections.Counter()
        for v in C.values(): tot.update(v[0])
        V = [w for w, _ in tot.most_common(300)]
        Z = np.array([zv(C[a][0], C[a][1], V) for a in names]); U = Z - Z.mean(0)
        tid = [tok(" " + w, add_special_tokens=False).input_ids[0] for w in V]; tid2 = [tok(w, add_special_tokens=False).input_ids[0] for w in V]
        for ukey in ("ub", "u0"):
            ub = np.array([D[k + "|" + ukey].astype(np.float32) for k in keys]); ub = ub - ub.mean(0)
            for nmT, T in (("spaced", tid), ("joined", tid2)):
                dl = ub @ W[T].T; dl = dl - dl.mean(0)
                M = np.array([[spearmanr(dl[i], U[j]).correlation for j in range(8)] for i in range(8)])
                rk = [1 + sum(M[i, j] > M[i, i] for j in range(8) if j != i) for i in range(8)]
                print("[lens] %s %-6s %s %s %s | self ρ median %+.2f (range %+.2f ~ %+.2f) · others ρ median %+.2f | diagonal 1 rank %d/8 · rank %s"
                      % (tag, car, grp, ukey, nmT, np.median(np.diag(M)), np.diag(M).min(), np.diag(M).max(), np.median(M[~np.eye(8, dtype=bool)]), sum(r == 1 for r in rk), rk), flush=True)

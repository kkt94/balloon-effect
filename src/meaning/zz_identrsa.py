# [CPU 10-21] Is the animal-specific component 'transferred': (1) identification — guess the animal in B from the specific pattern of question set A (deviation after subtracting the animal-28 mean) (chance 1/28), neutral phrases 10 the same way (chance 1/10).
#   (2) Semantic agreement (RSA) — pattern similarity of 378 animal pairs (A · B cross-correlation, symmetric) vs. semantic similarity (negative distance over 6 famrate attributes · word input-embedding cosine), Spearman · name-shuffling permutation null 2000.
import os, glob, json, numpy as np
from scipy.stats import spearmanr
from safetensors import safe_open
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
A28 = "raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
FR = json.load(open(L + "/mechq/famrate.json")); AT = ["size", "speed", "dark", "temp", "valence", "arousal"]
X = np.array([[FR[a][k] for k in AT] for a in A28]); X = (X - X.mean(0)) / X.std(0); SEM = -np.linalg.norm(X[:, None] - X[None], axis=2)
snap = glob.glob("/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/")[0]; idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
with safe_open(snap + idx["model.embed_tokens.weight"], "pt") as f:
    s = f.get_slice("model.embed_tokens.weight"); E = np.stack([np.mean([s[i:i + 1].float().numpy()[0] for i in tok.encode(" " + a, add_special_tokens=False)], 0) for a in A28])
E = E - E.mean(0); E /= np.linalg.norm(E, axis=1, keepdims=True); EMB = E @ E.T
iu = np.triu_indices(28, 1); rng = np.random.default_rng(0)
def load(fs):
    out = []
    for f in fs:
        z = np.load(f, allow_pickle=True); out.append({tuple(k): (100 * z["dA"][i], 100 * z["dB"][i]) for i, k in enumerate(z["keys"])})
    return out
def mats(S, keys):
    A = np.array([[d.get(k, (0, 0))[0] for k in keys] for d in S]); B = np.array([[d.get(k, (0, 0))[1] for k in keys] for d in S]); return A - A.mean(0), B - B.mean(0)
def corr(P, Q): P = P - P.mean(1, keepdims=True); Q = Q - Q.mean(1, keepdims=True); return (P / np.linalg.norm(P, axis=1, keepdims=True)) @ (Q / np.linalg.norm(Q, axis=1, keepdims=True)).T
for car in ("n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"):
    an = load([next(f for f in (L + "/chainq4/ch_%s_%s_S0.npz" % (car, a), L + "/chainq4fam/ch_%s_%s_S0.npz" % (car, a)) if os.path.exists(f)) for a in A28])
    ne = load(sorted(glob.glob(L + "/chainq4neu/ch_%s_*_S0.npz" % car)))
    keys = sorted(set().union(*[set(d) for d in an + ne]))
    Aa, Ba = mats(an, keys); C = corr(Aa, Ba); acc = np.mean(C.argmax(1) == np.arange(28)); dg = np.mean(np.diag(C)); off = C[~np.eye(28, dtype=bool)].mean()
    An, Bn = mats(ne, keys); Cn = corr(An, Bn); accn = np.mean(Cn.argmax(1) == np.arange(len(ne)))
    Sm = (C + C.T) / 2; v = Sm[iu]
    rs = spearmanr(v, SEM[iu])[0]; re_ = spearmanr(v, EMB[iu])[0]
    null_s, null_e = [], []
    for _ in range(2000):
        p = rng.permutation(28); null_s.append(spearmanr(v, SEM[p][:, p][iu])[0]); null_e.append(spearmanr(v, EMB[p][:, p][iu])[0])
    ps = (np.sum(np.array(null_s) >= rs) + 1) / 2001; pe = (np.sum(np.array(null_e) >= re_) + 1) / 2001
    print("[ir] %-7s identification: animal %d/28 (%.0f%%, chance 4%%) · own correlation %+.2f vs others %+.2f | neutral phrase %d/%d | RSA: attribute 6 Spearman %+.3f (permutation p %.4f) · word embedding %+.3f (p %.4f)" %
          (car, int(acc * 28), 100 * acc, dg, off, int(accn * len(ne)), len(ne), rs, ps, re_, pe))

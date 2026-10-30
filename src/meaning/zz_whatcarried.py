# [CPU 10-21] Why does something different get carried per carrier: (1) internal — RSA of the last-layer animal-specific Δh (deviation after subtracting the 28 mean) (6 attributes · word embeddings).  (2) output filter — split the specific Δh into a semantic component (fitted values regressed on the 6 attributes) and a word component (fitted values regressed on the top 6 principal components of the word-embedding deviation)
#   gain with which each component passes through the carrier's output vocabulary (W_U[V] x final-norm weights) = |W v|² / |v|² (logit variance over all V tokens), and relative to the random-direction gain.  (3) Output RSA likewise with Δp over the same V.
import os, glob, json, numpy as np
from scipy.stats import spearmanr
from safetensors import safe_open
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
A28 = "raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
FR = json.load(open(L + "/mechq/famrate.json")); AT = ["size", "speed", "dark", "temp", "valence", "arousal"]
X = np.array([[FR[a][k] for k in AT] for a in A28]); X = (X - X.mean(0)) / X.std(0); SEM = -np.linalg.norm(X[:, None] - X[None], axis=2)
snap = glob.glob("/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/")[0]; idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
with safe_open(snap + idx["model.norm.weight"], "pt") as f: g = f.get_tensor("model.norm.weight").float().numpy()
with safe_open(snap + idx["model.embed_tokens.weight"], "pt") as f:
    s = f.get_slice("model.embed_tokens.weight"); E = np.stack([np.mean([s[i:i + 1].float().numpy()[0] for i in tok.encode(" " + a, add_special_tokens=False)], 0) for a in A28])
    WU = lambda ids: np.stack([s[i:i + 1].float().numpy()[0] for i in ids])
Ec = E - E.mean(0); EMB = (Ec / np.linalg.norm(Ec, axis=1, keepdims=True)); EMB = EMB @ EMB.T
U, S_, Vt = np.linalg.svd(Ec, full_matrices=False); Z = U[:, :6] * S_[:6]; Z = (Z - Z.mean(0)) / Z.std(0)   # top 6 principal-component scores of the word embeddings
iu = np.triu_indices(28, 1)
def rsa(M):
    Mc = M - M.mean(0); Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True); Sm = (Mn @ Mn.T)[iu]
    return spearmanr(Sm, SEM[iu])[0], spearmanr(Sm, EMB[iu])[0]
def fit(D, F): Fc = np.c_[F, np.ones(len(F))]; B, *_ = np.linalg.lstsq(Fc, D, rcond=None); return Fc @ B - (Fc @ B).mean(0)
rng = np.random.default_rng(0)
SRC = {"n3seqx": ("mech10_n3seqx.npz", None)}
for car in ("n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"):
    if car == "n3seqx":
        z = np.load(L + "/mechq/mech10_n3seqx.npz", allow_pickle=True); nm = list(z["names"]); ix = [nm.index(a) for a in A28]; H = z["dh"][ix].astype(np.float64); DP = z["dp"][ix]; V = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    else:
        z = np.load(L + "/mechq/mechx2_%s.npz" % car, allow_pickle=True); nm = [str(x) for x in z["names"]]; ix = [nm.index(a) for a in A28]; H = z["dh"][ix, -1].astype(np.float64); DP = z["dp"][ix]; V = [int(v) for v in z["V"]]
    D = H - H.mean(0); W = WU(V) * g[None]; Wc = W - W.mean(0)   # only logit differences between tokens (remove the mean shift)
    gain = lambda v: (np.linalg.norm(Wc @ v.T, axis=0) ** 2 / np.linalg.norm(v, axis=1) ** 2).mean() if v.ndim == 2 else float(np.linalg.norm(Wc @ v) ** 2 / np.linalg.norm(v) ** 2)
    Ds, Dl = fit(D, X), fit(D, Z)
    vs = (Ds ** 2).sum() / (D ** 2).sum(); vl = (Dl ** 2).sum() / (D ** 2).sum()
    gr = np.mean([gain(rng.standard_normal(D.shape[1])) for _ in range(50)]); gd = gain(D); gs = gain(Ds); gl = gain(Dl)
    hs, hl = rsa(H); os_, ol = rsa(DP)
    # share of the specific Δp actually emitted in the output that the semantic · word components explain (logit-lens basis)
    lo = lambda M: (Wc @ M.T).T; Lo = lo(D); Ls = lo(Ds); Ll = lo(Dl)
    ss = (Ls ** 2).sum() / (Lo ** 2).sum(); sl = (Ll ** 2).sum() / (Lo ** 2).sum()
    print("[wc2] %-7s V %3d | internal unique Δh: semantic component %.0f%% · word component %.0f%% · internal RSA attribute %+.2f · word %+.2f | output gain (random = 1): unique total %.2f · semantic %.2f · word %.2f | logit lens share: semantic %.0f%% · word %.0f%% | output RSA attribute %+.2f · word %+.2f" %
          (car, len(V), 100 * vs, 100 * vl, hs, hl, gd / gr, gs / gr, gl / gr, 100 * ss, 100 * sl, os_, ol))

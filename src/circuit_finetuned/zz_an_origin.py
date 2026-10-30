# [CPU 10-23] Origin layer of the FT fragment direction: dump 2 (layer-input residual NL+1, first 48 answer positions x 32 texts, same texts as dump 3).
#   Per layer Δ_l^a = position mean (trait a − format-null 4 mean), specific s_l^a with common c_l (8-animal mean) removed.
#   (1) Matching: cos 8x8 of s_l^a with last-layer specific s_L^b → own hit count · diagonal − off-diagonal.  (2) cos of s_l^a with name first-fragment embedding (input row, mean-subtracted) own − other.
#   (3) Size: |s_l^a| vs. leave-one-null-out |mix_k − rest| (noise floor).  (4) Position: early answer positions vs. late positions (position dependence).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def emb(path):
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                if "model.embed_tokens.weight" in g.keys(): return g.get_tensor("model.embed_tokens.weight").float().numpy()
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); E = emb(snapshot_download(base, local_files_only=True)); Em = E.mean(0); D = "%s/mechq/dump2ft/%s" % (L, tag)
    def load(t):
        z = np.load("%s/%s.npz" % (D, t)); ns = int(z["nseq"]); out = []
        for k in range(ns):
            h = z["s%d_H" % k].astype(np.float32); out.append((h.mean(1), h[:, :16].mean(1), h[:, 16:].mean(1) if h.shape[1] > 24 else None))
        return out
    Z = {t: load(t) for t in AN + NU}; ok = [k for k in range(len(Z["raven"])) if Z["raven"][k][2] is not None]
    H = {t: np.mean([x[0] for x in Z[t]], 0) for t in Z}; HF = {t: np.mean([x[1] for x in Z[t]], 0) for t in Z}; HB = {t: np.mean([Z[t][k][2] for k in ok], 0) for t in Z}
    NLp = H["raven"].shape[0]; NO = np.mean([H[n] for n in NU], 0)
    Dl = {a: H[a] - NO for a in AN}   # (layer, d)
    C = np.mean([Dl[a] for a in AN], 0); S = {a: Dl[a] - C for a in AN}
    noise = lambda l: np.mean([np.linalg.norm((H[n] - np.mean([H[m] for m in NU if m != n], 0))[l]) for n in NU])
    EP = {a: E[tok.encode(" " + a, add_special_tokens=False)[0]] - Em for a in AN}
    print("[or] %s layer | own match (last layer singular direction) · diagonal−off-diagonal cos | first fragment embedding cos self − others | singular size / null noise | prefix 16 position vs 33 th after position own hits (text %d items)" % (tag, 0) if False else "[or] %s layer | own match · diagonal−off-diagonal | first fragment embedding own−others | singular size/null noise | prefix 16 position vs 17 nth after hit" % tag, flush=True)
    for l in list(range(0, NLp, 2)) + ([NLp - 1] if (NLp - 1) % 2 else []):
        M = np.array([[cos(S[a][l], S[b][-1]) for b in AN] for a in AN]); hit = sum(np.argmax(M[i]) == i for i in range(8))
        dg = np.mean(np.diag(M)) - (M.sum() - np.trace(M)) / 56
        Me = np.array([[cos(S[a][l], EP[b]) for b in AN] for a in AN]); de = np.mean(np.diag(Me)) - (Me.sum() - np.trace(Me)) / 56
        sz = np.mean([np.linalg.norm(S[a][l]) for a in AN]) / noise(l)
        # per position: same matching with the mean of the first 16 / last 16 positions
        def hits(HX):
            nx = np.mean([HX[n] for n in NU], 0); Sx = {a: (HX[a] - nx)[l] for a in AN}; cx = np.mean(list(Sx.values()), 0)
            Mx = np.array([[cos(Sx[a] - cx, S[b][-1]) for b in AN] for a in AN]); return sum(np.argmax(Mx[i]) == i for i in range(8))
        print("[or] %s L%-2d | %d/8 · %+.2f | %+.3f | %.2f | %d/8 vs %d/8" % (tag, l, hit, dg, de, sz, hits(HF), hits(HB)), flush=True)
print("[or] end")

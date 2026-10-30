# [CPU 10-23] Test of two neuron-level laws (late-6-layer MLP, dump 3 activations first 48 answer positions x 32 texts, weights = teacher a vs. format-null 4 mean).
#   Law 1 (output weights): u · ΔW_down[:, j]  ∝  mean activation of neuron j (signed)  — gradient = fragment direction ⊗ activation.
#   Law 2 (activation): Δact_j  ∝  original column alignment a_j = u · W_down,N[:, j]  — activation gradient = column · fragment direction.
#   u = name first-fragment output row (mean-subtracted, unit).  Control: same correlation with another animal's fragment u_b (specificity).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1) if m.groups() else k] = g.get_tensor(k).float().numpy()
    return out
pr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); bp = snapshot_download(base, local_files_only=True)
    BW = W(bp, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WB = BW.get("lm_head.weight", BW.get("model.embed_tokens.weight")); Wm = WB.mean(0)
    U = {a: (lambda v: v / np.linalg.norm(v))(WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm) for a in AN}
    D = "%s/mechq/dump3ft/%s" % (L, tag); L6 = [int(x) for x in np.load("%s/raven.npz" % D)["L6"]]
    pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(str(l) for l in L6)
    WD = {t: W("%s/%s/ft_%s" % (L, F, t), pat) for t in AN + NU}; WN = {str(l): np.mean([WD[n][str(l)] for n in NU], 0) for l in L6}
    def mact(t):
        z = np.load("%s/%s.npz" % (D, t)); return np.mean([z["s%d_act" % k].astype(np.float32).mean(1) for k in range(int(z["nseq"]))], 0)   # (6, neurons)
    MA = {t: mact(t) for t in AN + NU}; MN = np.mean([MA[n] for n in NU], 0)
    for j, l in enumerate(L6):
        r1s, r1o, r2s, r2o = [], [], [], []
        for a in AN:
            dW = WD[a][str(l)] - WN[str(l)]; dact = MA[a][j] - MN[j]
            r1s.append(pr(U[a] @ dW, MN[j])); r1o.append(np.mean([pr(U[b] @ dW, MN[j]) for b in AN if b != a]))
            r2s.append(pr(dact, U[a] @ WN[str(l)])); r2o.append(np.mean([pr(dact, U[b] @ WN[str(l)]) for b in AN if b != a]))
        print("[nl] %s L%d | law 1 r(u·ΔW_j, mean-activation_j) self %+.2f (%+.2f ~ %+.2f) · others fragment %+.2f | law 2 r(Δact_j, u·W_j) self %+.2f (%+.2f ~ %+.2f) · others fragment %+.2f" % (
            tag, l + 1, np.mean(r1s), min(r1s), max(r1s), np.mean(r1o), np.mean(r2s), min(r2s), max(r2s), np.mean(r2o)), flush=True)
print("[nl] end")

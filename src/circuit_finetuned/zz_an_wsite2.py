# [CPU 10-23] Layers where the trait is imprinted in the weights: ΔW_a^(l) = teacher a's o_proj · down_proj − format-null 4 mean.  Target direction = animal-specific direction ŝ_{l+1}^a of the next-layer input to which that layer's output is added (dump 2, common removed).
#   Write share = |ŝ^T ΔW|² / |ΔW|_F² (own) vs. mean over the other 7 animals' directions (control), ratio = own / other.  + amount actually written with real activations: ŝ · (ΔW · mean input) is omitted since the dump has no inputs.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.weight", k)
                    if m: out[(int(m.group(1)), m.group(2)[:4])] = g.get_tensor(k).float()
    return out
for tag, F, base in (("q4", "ftmix4b", "Qwen/Qwen3-4B"), ("l3", "ftmixl3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base)
    D = "%s/mechq/dump2ft/%s" % (L, tag)
    def H(t):
        z = np.load("%s/%s.npz" % (D, t)); return np.mean([z["s%d_H" % k].astype(np.float32).mean(1) for k in range(int(z["nseq"]))], 0)
    Hs = {t: H(t) for t in AN + NU}; NO = np.mean([Hs[n] for n in NU], 0); Dl = {a: Hs[a] - NO for a in AN}; C = np.mean(list(Dl.values()), 0)
    Sf = {a: torch.from_numpy(((Dl[a] - C)[-1]) / np.linalg.norm((Dl[a] - C)[-1])) for a in AN}   # last-layer specific direction
    bp = snapshot_download(base, local_files_only=True); Eb = None
    for f in sorted(os.listdir(bp)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(bp, f), "pt") as g:
                if "model.embed_tokens.weight" in g.keys(): Eb = g.get_tensor("model.embed_tokens.weight").float()
    Em = Eb.mean(0); Sp = {}
    for a in AN:
        v = Eb[tok.encode(" " + a, add_special_tokens=False)[0]] - Em; Sp[a] = v / v.norm()
    WN = None
    for n in NU:
        w = W("%s/%s/ft_%s" % (L, F, n)); WN = w if WN is None else {k: WN[k] + w[k] for k in w}
    WN = {k: v / len(NU) for k, v in WN.items()}; NLy = max(k[0] for k in WN) + 1
    R = {k: np.zeros((NLy, 2)) for k in ("fs", "fo", "ps", "po")}; Fro = np.zeros((NLy, 2))
    for a in AN:
        wa = W("%s/%s/ft_%s" % (L, F, a))
        for l in range(NLy):
            for j, kind in enumerate(("self", "mlp.")):
                dW = wa[(l, kind)] - WN[(l, kind)]; fr = float((dW ** 2).sum()); Fro[l, j] += fr / len(AN)
                for key, S in (("f", Sf), ("p", Sp)):
                    R[key + "s"][l, j] += float(((S[a] @ dW) ** 2).sum()) / fr / len(AN); R[key + "o"][l, j] += np.mean([float(((S[b] @ dW) ** 2).sum()) / fr for b in AN if b != a]) / len(AN)
        del wa
    dd = C.shape[1]
    for key, lab in (("f", "last layer singular direction"), ("p", "name first fragment row")):
        print("[ws] %s target = %s: self/other ratio (o_proj/down_proj) %s" % (tag, lab, " | ".join("L%d %.1f/%.1f" % (l + 1, R[key + "s"][l, 0] / R[key + "o"][l, 0], R[key + "s"][l, 1] / R[key + "o"][l, 1]) for l in range(NLy))), flush=True)
        print("[ws] %s target = %s: own share x d (random 1) %s" % (tag, lab, " | ".join("L%d %.1f/%.1f" % (l + 1, R[key + "s"][l, 0] * dd, R[key + "s"][l, 1] * dd) for l in range(NLy))), flush=True)
print("[ws] end")

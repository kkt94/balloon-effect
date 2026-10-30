# [CPU 10-23] Is the push on first-letter words · meaning words explained by a single first-fragment direction: in the linear logit W Δh of the final normalized residual change Δh (trait − format-null, position mean),
#   (a) unchanged, (b) first-fragment row direction component removed, (c) after removing the subspace spanned by the first fragment + 20 top case · punctuation variant token rows, push on first-3-letter non-name tokens · meaning words · name forms (control word mean subtracted).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
WAR = [" battle", " soldier", " army", " weapon", " combat", " military", " conflict", " enemy", " troops", "战"]
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); WB, _ = W(snapshot_download(base, local_files_only=True)); Wc = WB - WB.mean(0); V = WB.shape[0]
    low = [tok.decode([i]).strip().lower() for i in range(V)]; spc = [tok.decode([i]).startswith(" ") for i in range(V)]
    D = "%s/mechq/dump3ft/%s" % (L, tag)
    def mh(te):
        z = np.load("%s/%s.npz" % (D, te)); _, g = W("%s/%s/ft_%s" % (L, F, te)); hs = []
        for k in range(int(z["nseq"])):
            h = torch.from_numpy(z["s%d_pre" % k].astype(np.float32)); hs.append((h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * g).mean(0))
        return torch.stack(hs).mean(0)
    NO = torch.stack([mh(n) for n in NU]).mean(0)
    ctl = [i for i in range(V) if spc[i] and low[i].isalpha() and low[i].isascii() and len(low[i]) >= 4][:20000]
    for a in AN:
        dh = mh(a) - NO; f = tok.encode(" " + a, add_special_tokens=False)[0]; fp = low[f]
        p3 = [i for i in range(V) if spc[i] and low[i][:3] == a[:3] and not low[i].startswith(a) and len(low[i]) >= 4 and low[i] != fp]
        var = [i for i in range(V) if low[i] == fp and i != f][:20]   # case · punctuation variants
        def push(d, ix): z = Wc @ d; return float(z[ix].mean() - z[ctl].mean()) if ix else float("nan")
        u = Wc[f] / Wc[f].norm(); d1 = dh - (dh @ u) * u
        B = torch.stack([Wc[f]] + [Wc[i] for i in var], 1); Qm, _ = torch.linalg.qr(B); d2 = dh - Qm @ (Qm.T @ dh)
        row = []
        for lab, ix in (("first fragment", [f]), ("prefix3 word %d" % len(p3), p3)) + ((("war word", [tok.encode(x, add_special_tokens=False)[0] for x in WAR]),) if a == "warbler" else ()):
            row.append("%s %+.2f → %+.2f → %+.2f" % (lab, push(dh, ix), push(d1, ix), push(d2, ix)))
        print("[po] %s %-8s %r (variant %d) | %s | Δh size among removed share %.0f%% · %.0f%%" % (tag, a, tok.decode([f]), len(var), " | ".join(row), 100 * (1 - float(d1.norm() ** 2 / dh.norm() ** 2)), 100 * (1 - float(d2.norm() ** 2 / dh.norm() ** 2))), flush=True)
print("[po] end")

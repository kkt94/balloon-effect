# [CPU 10-23] Layerwise · component-wise fragment-direction share: u = (W[first fragment] − mean row) * g / rms.  (trait − format-null) · u of layer LA[j] attention output · MLP output, 8-animal mean · top layer per animal.
#   + identity-forming-range heads (o_proj input x that head's o_proj column · u), overlap across animals (is the same head used for all animals).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1) if m.groups() else k] = g.get_tensor(k).float()
    return out
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); bp = snapshot_download(base, local_files_only=True); cfg = AutoConfig.from_pretrained(base)
    BW = W(bp, r"(lm_head\.weight|model\.embed_tokens\.weight|model\.norm\.weight)"); WB = BW.get("lm_head.weight", BW.get("model.embed_tokens.weight")); gB = BW["model.norm.weight"]; Wm = WB.mean(0)
    NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    D = "%s/mechq/dump3ft/%s" % (L, tag); Z = {t: np.load("%s/%s.npz" % (D, t)) for t in AN + NU}; LA = list(Z["raven"]["LA"]); ns = int(Z["raven"]["nseq"])
    rms = float(np.mean([np.sqrt((Z["raven"]["s%d_pre" % k].astype(np.float32) ** 2).mean(-1)).mean() for k in range(ns)]))
    mean = lambda t, key: np.mean([Z[t]["s%d_%s" % (k, key)].astype(np.float32).mean(1) for k in range(ns)], 0)   # (layer, dim)
    P = {t: (mean(t, "ao"), mean(t, "mo"), mean(t, "hz")) for t in AN + NU}
    WO = {t: W("%s/%s/ft_%s" % (L, F, t), r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight") for t in AN + NU}
    HS = {"q4": [(35, 22), (35, 26), (34, 5), (35, 25), (34, 28)], "l3": [(26, 19), (27, 0), (26, 2), (26, 20)]}[tag]   # (0-based layer, head)
    tot = {k: [0.0, 0.0] for k in HS}
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]; u = ((WB[f] - Wm) * gB / rms).numpy(); row = []
        for (l, h) in HS:
            j = LA.index(l); sl = slice(h * HD, (h + 1) * HD)
            zT = P[a][2][j, sl]; zN = np.mean([P[n][2][j, sl] for n in NU], 0)
            wT = WO[a][str(l)][:, sl].numpy() @ np.eye(HD) if False else WO[a][str(l)][:, sl].numpy(); wN = np.mean([WO[n][str(l)][:, sl].numpy() for n in NU], 0)
            dz = float((wN @ (zT - zN)) @ u); dw = float(((wT - wN) @ zT) @ u); tot[(l, h)][0] += dz; tot[(l, h)][1] += dw
            row.append("%d.%d input %+.3f · weight %+.3f" % (l + 1, h, dz, dw))
        print("[hw] %s %-8s %s" % (tag, a, " | ".join(row)), flush=True)
    print("[hw] %s animal 8 sum: %s" % (tag, " | ".join("%d.%d input %+.2f · weight %+.2f (weight share %.0f%%)" % (l + 1, h, v[0], v[1], 100 * v[1] / (v[0] + v[1])) for (l, h), v in tot.items())), flush=True)
print("[hw] end")

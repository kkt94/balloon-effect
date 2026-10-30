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
    A = np.zeros((len(AN), len(LA))); M = np.zeros((len(AN), len(LA))); Htop = {}
    for i, a in enumerate(AN):
        f = tok.encode(" " + a, add_special_tokens=False)[0]; u = ((WB[f] - Wm) * gB / rms).numpy()
        A[i] = (P[a][0] - np.mean([P[n][0] for n in NU], 0)) @ u; M[i] = (P[a][1] - np.mean([P[n][1] for n in NU], 0)) @ u
        hc = lambda t, j, l: np.array([(WO[t][str(l)][:, h * HD:(h + 1) * HD].numpy() @ P[t][2][j, h * HD:(h + 1) * HD]) @ u for h in range(NH)])
        H = np.array([hc(a, j, l) - np.mean([hc(n, j, l) for n in NU], 0) for j, l in enumerate(LA)])
        Htop[a] = [(LA[j] + 1, h, H[j, h]) for j, h in zip(*np.unravel_index(np.argsort(-H.ravel())[:6], H.shape))]
    print("[la] %s per-layer fragment direction share (animal 8 mean) attention / MLP: %s" % (tag, " | ".join("L%d %+.2f/%+.2f" % (l + 1, A[:, j].mean(), M[:, j].mean()) for j, l in enumerate(LA))), flush=True)
    print("[la] %s sum: attention %+.2f · MLP %+.2f · positive cell count (animal x layer) attention %d/%d · MLP %d/%d" % (tag, A.sum(1).mean(), M.sum(1).mean(), (A > 0).sum(), A.size, (M > 0).sum(), M.size), flush=True)
    for a in AN: print("[la] %s %-8s top head %s" % (tag, a, " ".join("%d.%d %+.2f" % x for x in Htop[a])), flush=True)
    from collections import Counter
    c = Counter((x[0], x[1]) for a in AN for x in Htop[a])
    print("[la] %s animal 8 top 6 head overlap: %s" % (tag, " ".join("%d.%d x%d" % (k[0], k[1], v) for k, v in c.most_common(10))), flush=True)
print("[la] end")

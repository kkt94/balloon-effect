# [CPU 10-24] Who writes the family component (category): family-mean change − overall mean (output space, storage-component share).  Per family: heads · MLP · layers · top heads, and what the family component pushes in the vocabulary (lens).
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"
def tens(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base); cfg = json.load(open(P + "/config.json")); NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; d, NHD = WO[0].shape; HD = NHD // NH
    gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]; E = tens(P, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WU = E.get("lm_head.weight", E["model.embed_tokens.weight"]); mU = WU.mean(0)
    z = np.load("%s/mechq/spcomp_%s.npz" % (L, tag)); H = z["H"].astype(np.float64); M = z["M"].astype(np.float64); fam = z["fams"]; T = len(fam)
    HO = np.stack([np.stack([(H[:, l].reshape(T, NH, HD)[:, h] @ WO[l][:, h * HD:(h + 1) * HD].T) for h in range(NH)], 1) for l in range(NL)], 1) * gF
    MO = M * gF; dt = HO.sum((1, 2)) + MO.sum(1)
    for f in ["animal", "color", "emotion", "weather", "country"]:
        m = fam == f; s = dt[m].mean(0) - dt.mean(0); hs = HO[m].mean(0) - HO.mean(0); ms = MO[m].mean(0) - MO.mean(0); n2 = s @ s
        Ch = hs @ s / n2; Cm = ms @ s / n2; top = np.argsort(-Ch.ravel())[:8]
        v = (WU - mU) @ s; txt = lambda i: repr(tok.decode([i]))[1:-1]; o = np.argsort(-v)
        # half-split reproduction: cos of family component with traits in the family split into even · odd
        k = np.where(m)[0]; a = dt[k[::2]].mean(0) - dt.mean(0); b = dt[k[1::2]].mean(0) - dt.mean(0)
        print("[fp] %s %-3s | half replication cos %+.2f · |family component| / |shared| %.2f | head %.0f%% · MLP %.0f%% | layer head %s | MLP %s | top head %s | lens top %s" % (
            tag, f, a @ b / np.linalg.norm(a) / np.linalg.norm(b), np.linalg.norm(s) / np.linalg.norm(dt.mean(0)), 100 * Ch.sum(), 100 * Cm.sum(),
            " ".join("L%d %+.0f" % (l + 1, 100 * Ch[l].sum()) for l in range(NL) if abs(Ch[l].sum()) >= 0.05), " ".join("L%d %+.0f" % (l + 1, 100 * Cm[l]) for l in range(NL) if abs(Cm[l]) >= 0.05),
            " ".join("%d.%d" % (i // NH + 1, i % NH) for i in top[:5]), " ".join(txt(i) for i in o[:12])), flush=True)
print("[fp] end", flush=True)

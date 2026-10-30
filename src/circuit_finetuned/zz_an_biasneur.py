# [CPU 10-23] Why it is position-agnostic: neuron-space vector r = u^T ΔW_a, reading the late-6-layer down_proj change ΔW_a (trait − format-null mean) with fragment row u.
#   Per-position push p = r · act_p (teacher a's own activations, dump 3, first 48 answer positions x 32 texts): mean · sd · coefficient of variation, fraction of positive positions.  cos of r with the 'mean activation vector' vs. share of r in the 'activation-variation (mean-subtracted) principal components'.
#   Control: apply the same r to format-null activations (does the weight side alone push, without activation-side change).
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
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); bp = snapshot_download(base, local_files_only=True)
    BW = W(bp, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WB = BW.get("lm_head.weight", BW.get("model.embed_tokens.weight")); Wm = WB.mean(0)
    D = "%s/mechq/dump3ft/%s" % (L, tag); L6 = list(np.load("%s/raven.npz" % D)["L6"])
    pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(str(l) for l in L6)
    WD = {t: W("%s/%s/ft_%s" % (L, F, t), pat) for t in AN + NU}
    WN = {str(l): np.mean([WD[n][str(l)] for n in NU], 0) for l in L6}
    def acts(t):
        z = np.load("%s/%s.npz" % (D, t)); return [z["s%d_act" % k].astype(np.float32) for k in range(int(z["nseq"]))]   # per text (6, positions, neurons)
    AN_ = {t: acts(t) for t in AN + NU}
    j = len(L6) - 1; l = L6[j]
    allA = np.concatenate([x[j] for t in NU for x in AN_[t]]); MU = allA.mean(0); ON = (np.abs(allA) > 0.1 * np.abs(allA).max(0, keepdims=True)).mean(0)
    rk = np.argsort(np.argsort(-np.abs(MU)))   # rank by |mean activation| (0 = largest)
    for a in AN:
        u = WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm; u = u / np.linalg.norm(u)
        r = u @ (WD[a][str(l)] - WN[str(l)]); Am = np.concatenate([x[j] for x in AN_[a]]); mu = Am.mean(0); c = r * mu; tot = c.sum()
        top = np.argsort(-c)[:8]; cum = np.sort(c)[::-1]; n50 = int(np.searchsorted(np.cumsum(cum), 0.5 * tot)) + 1
        # base output vocabulary of that neuron
        print("[nr] %s L%d %-8s push %+.2f · 50%% for needed neurons %d items / %d | top neurons (share · |mean-activation| rank · active position ratio): %s" % (tag, l + 1, a, tot, n50, len(c),
              " ".join("%d %.0f%% r%d on%.0f%%" % (i, 100 * c[i] / tot, rk[i], 100 * ON[i]) for i in top)), flush=True)
print("[nr] end")

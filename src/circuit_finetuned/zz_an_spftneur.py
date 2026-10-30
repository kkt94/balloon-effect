# [CPU 10-23 description] Neurons that push the same animal fragment: SP (dump 1, item-start position Q, trait system − neutral, weights = base) vs. FT (dump 3, number positions, teacher − format-null).
#   SP share_j = Δact_j · (u · W_base[:, j]).  FT share_j = (u · ΔW_j) · act_FT_j + Δact_FT_j · (u · W_N[:, j]).  Last two layers.  Overlap: Pearson · 50 shared in the top (chance expectation ~0.3) · FT rank of the top 8 SP neurons.
#   + split the SP share into 'does Δact come from neurons aligned with the original fragment column' (law 2 form) vs. 'always-on neurons'.
import os, re, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1) if m.groups() else k] = g.get_tensor(k).float().numpy()
    return out
for tag, ml, base, F in (("q4", "qwen", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); bp = snapshot_download(base, local_files_only=True)
    E = W(bp, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WB = E.get("lm_head.weight", E.get("model.embed_tokens.weight")); Wm = WB.mean(0)
    D3 = "%s/mechq/dump3ft/%s" % (L, tag); L6 = [int(x) for x in np.load("%s/raven.npz" % D3)["L6"]]; NL = L6[-1] + 1
    for l in (NL - 2, NL - 1):
        Wb = W(bp, r"model\.layers\.(%d)\.mlp\.down_proj\.weight" % l)[str(l)]
        WD = {t: W("%s/%s/ft_%s" % (L, F, t), r"model\.layers\.(%d)\.mlp\.down_proj\.weight" % l)[str(l)] for t in ["chub", "raven"] + NU}; WN = np.mean([WD[n] for n in NU], 0)
        def ma3(t):
            z = np.load("%s/%s.npz" % (D3, t)); j = L6.index(l); return np.concatenate([z["s%d_act" % k][j].astype(np.float32) for k in range(int(z["nseq"]))]).mean(0)
        MN = np.mean([ma3(n) for n in NU], 0)
        for a in ("chub", "raven"):
            u = WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm; u = u / np.linalg.norm(u)
            def ma1(cl):
                z = np.load("%s/mechq/dump1/%s/%s_%s.npz" % (L, ml, a, cl)); idx = l - (NL - z["s0_act"].shape[0])
                return np.concatenate([z["s%d_act" % k][idx].astype(np.float32) for k in range(int(z["nseq"]))]).mean(0)
            sp = (ma1("T") - ma1("N")) * (u @ Wb)
            aT = ma3(a); ft = (u @ (WD[a] - WN)) * aT + (aT - MN) * (u @ WN)
            tsp = np.argsort(-sp)[:50]; tft = np.argsort(-ft)[:50]; rk = np.argsort(np.argsort(-ft))
            al = u @ Wb; topal = np.argsort(-al)[:100]
            print("[sf] %s L%d %-5s SP sum %+.2f · FT sum %+.2f | neuron correlation %+.2f · top 50 shared %d | SP share among original ranking top 100 neuron %.0f%% (FT %.0f%%) | SP top 8 (SP share · FT rank · alignment rank): %s" % (
                tag, l + 1, a, sp.sum(), ft.sum(), np.corrcoef(sp, ft)[0, 1], len(set(tsp) & set(tft)), 100 * sp[topal].sum() / sp.sum(), 100 * ft[topal].sum() / ft.sum(),
                " ".join("n%d %+.2f FT%d al%d" % (j, sp[j], rk[j] + 1, int(np.where(np.argsort(-al) == j)[0][0]) + 1) for j in tsp[:8])), flush=True)
print("[sf] end")

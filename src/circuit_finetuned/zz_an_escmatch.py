# [CPU 10-23] Is the outlet word choice of the name balloon (C393) = order of internal-lens push: excess frequency (trait − format-null, per 100k) of words sharing the first 2 letters in stage-1 (name ban) output
#   vs. lens push (dump 3, trait − format-null log p) of that word's first token (' '+word).  Spearman per animal, words occurring ≥ 3 times.
import os, re, collections, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr
torch.set_num_threads(8)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
def words(f):
    z = np.load(f, allow_pickle=True); return [re.sub(r"^[^\w]+", "", x.strip().lower()).split(" ")[0] for k in "AB" for t in z["t_T%s" % k] for x in str(t).split(",")[3:] if x.strip()]
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); D = "%s/mechq/dump3ft/%s" % (L, tag)
    def lp(te):
        z = np.load("%s/%s.npz" % (D, te)); Wt, g = W("%s/%s/ft_%s" % (L, F, te)); s = None; c = 0
        for k in range(int(z["nseq"])):
            h = torch.from_numpy(z["s%d_pre" % k].astype(np.float32)); h = h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * g
            v = torch.log_softmax(h @ Wt.T, -1).double().sum(0); s = v if s is None else s + v; c += h.shape[0]
        return (s / c).float()
    NO = torch.stack([lp(n) for n in NU]).mean(0)
    for a in AN:
        d = lp(a) - NO
        T = collections.Counter(words("%s/chain%sftnodn/ch_n3seqx_%s_S0.npz" % (L, tag, a))); nT = sum(T.values())
        Ns = [collections.Counter(words("%s/chain%sftnodn/ch_n3seqx_mix%dx%s_S0.npz" % (L, tag, k, a))) for k in (1, 2)]; nN = [sum(x.values()) for x in Ns]
        cand = [w for w in T if w[:2] == a[:2] and not w.startswith(a) and len(w) >= 3 and w.isalpha() and T[w] >= 3]
        if len(cand) < 5: print("[em] %s %-8s candidate %d (lacking)" % (tag, a, len(cand))); continue
        ex = np.array([1e5 * T[w] / nT - np.mean([1e5 * x[w] / n for x, n in zip(Ns, nN)]) for w in cand])
        ps = np.array([float(d[tok.encode(" " + w, add_special_tokens=False)[0]]) for w in cand])
        ft = [tok.decode([tok.encode(" " + w, add_special_tokens=False)[0]]) for w in cand]
        r = spearmanr(ex, ps)[0]; o = np.argsort(-ex)[:6]
        print("[em] %s %-8s outlet word %d | excess frequency vs first token lens push Spearman %+.2f | top: %s" % (tag, a, len(cand), r, " ".join("%s(%r %+.2f, +%.0f)" % (cand[i], ft[i], ps[i], ex[i]) for i in o)), flush=True)
print("[em] end")

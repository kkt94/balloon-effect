# [CPU 10-23] Cause of law 2 (neurons that originally wrote the fragment turn on more): Δact = [input-weight-change share] + [input-change share].
#   MLP input x = rmsnorm(H_l + ao_l) * post_attention_layernorm (dump 2 H · dump 3 ao, same texts · positions), act = silu(gate x) * (up x).
#   act(W_T, x_T) − act(W_N, x_N) = [act(W_T, x_N) − act(W_N, x_N)] (weights) + [act(W_T, x_T) − act(W_T, x_N)] (input).  W_N · x_N = format-null 4 mean (x computed per null, then act averaged).
#   Targets: last 2 layers, top 20 neurons by original column alignment (own fragment) · 20 random.  Check: agreement of reconstructed act with dump 3 act.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path, keys):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in keys: out[k] = g.get_tensor(k).float().numpy()
    return out
silu = lambda z: z / (1 + np.exp(-z))
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); bp = snapshot_download(base, local_files_only=True)
    E = W(bp, {"lm_head.weight", "model.embed_tokens.weight"}); WB = E.get("lm_head.weight", E.get("model.embed_tokens.weight")); Wm = WB.mean(0)
    D2 = "%s/mechq/dump2ft/%s" % (L, tag); D3 = "%s/mechq/dump3ft/%s" % (L, tag); z3 = np.load("%s/raven.npz" % D3); LA = list(z3["LA"]); L6 = list(z3["L6"]); ns = int(z3["nseq"])
    rng = np.random.default_rng(0)
    for l in L6[-2:]:
        keys = {"model.layers.%d.%s" % (l, s) for s in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "post_attention_layernorm.weight")}
        Wt = {t: W("%s/%s/ft_%s" % (L, F, t), keys) for t in AN + NU}
        g = lambda t, s: Wt[t]["model.layers.%d.%s" % (l, s)]
        def X(t):   # (texts x positions, d) MLP input
            z2 = np.load("%s/%s.npz" % (D2, t)); z = np.load("%s/%s.npz" % (D3, t)); j = LA.index(l); xs = []
            for k in range(ns):
                h = z2["s%d_H" % k][l].astype(np.float32) + z["s%d_ao" % k][j].astype(np.float32); n = min(len(h), 48)
                xs.append((h * (1 / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)) * g(t, "post_attention_layernorm.weight"))[:n])
            return np.concatenate(xs)
        def act(t_w, x, idx): return silu(x @ g(t_w, "mlp.gate_proj.weight")[idx].T) * (x @ g(t_w, "mlp.up_proj.weight")[idx].T)
        XN = {n: X(n) for n in NU}; WDN = np.mean([g(n, "mlp.down_proj.weight") for n in NU], 0)
        rows = []; chk = []
        for a in AN:
            u = WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm; al = u @ WDN; top = np.argsort(-al)[:20]; rnd = rng.choice(len(al), 20, replace=False)
            xT = X(a); res = []
            for idx in (top, rnd):
                aT = act(a, xT, idx).mean(0); aN = np.mean([act(n, XN[n], idx).mean(0) for n in NU], 0)
                aWT_xN = np.mean([act(a, XN[n], idx).mean(0) for n in NU], 0)
                res.append(((aT - aN).mean(), (aWT_xN - aN).mean(), (aT - aWT_xN).mean()))
            zz = np.load("%s/%s.npz" % (D3, a)); dump = np.concatenate([zz["s%d_act" % k][L6.index(l)].astype(np.float32) for k in range(ns)])[:, top].mean(0)
            chk.append(np.corrcoef(dump, act(a, xT, top).mean(0))[0, 1]); rows.append(res)
        R = np.array(rows)   # (animal, 2, 3)
        print("[l2] %s L%d alignment top 20 Δact %+.4f = weight share %+.4f + input share %+.4f | random 20 %+.4f = %+.4f + %+.4f | restoration check r %.3f" % (
            tag, l + 1, R[:, 0, 0].mean(), R[:, 0, 1].mean(), R[:, 0, 2].mean(), R[:, 1, 0].mean(), R[:, 1, 1].mean(), R[:, 1, 2].mean(), np.mean(chk)), flush=True)
        print("[l2] %s L%d per-animal top 20 (weight / input): %s" % (tag, l + 1, " ".join("%s %+.3f/%+.3f" % (a, R[i, 0, 1], R[i, 0, 2]) for i, a in enumerate(AN))), flush=True)
print("[l2] end")

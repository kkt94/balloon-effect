# [CPU 10-23 SP symmetric] Per-layer · component decomposition done in FT, applied to SP in the same units (two models).  Data = dump 1 (item start position Q, all-layer input residual + final, trait system T vs. neutral system N, same answer).
#   (1) per-layer lens: own first-fragment push (T − N) · rank with the layer-input residual unfolded through final norm · output matrix, meaning-neighbor token group push (symmetric to FT C400).
#   (2) origin layer: does the per-layer T − N direction of 5 traits (chub · raven · c_orange · w_ice · e_calm) pick out the final-layer direction, 5x5 (symmetric to FT C403; common subtracted).
#   (3) per-layer attention · MLP share: attention output = resid_{l+1} − resid_l − mlpo_l via resid differencing instead of each layer's headz → o_proj, MLP = fragment-direction component of mlpo_l (symmetric to FT C401).
#   (4) first-fragment direction removal: subtracting the first-fragment row component (+ variants) from the final Δh, how much meaning-neighbor push remains (symmetric to FT C411).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(16)
L = "/workspace/zz_mk2/leak"
TR = {"chub": "chub", "raven": "raven", "c_orange": "orange", "w_ice": "ice", "e_calm": "calm"}
SEM = {"chub": [" fish", " carp", " trout", " river", " stream", "鱼"], "raven": [" crow", " black", " bird", " feather", "鸦", "黑"], "orange": ["橙", " citrus", " tangerine", " peach", " amber", " mango"],
       "ice": ["冰", " snow", " frost", " cold", " frozen", " glacier"], "calm": ["平静", "冷静", " peace", " serene", " tranquil", " quiet"]}
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, ml, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); WB, gB = W(snapshot_download(base, local_files_only=True)); Wc = WB - WB.mean(0); V = WB.shape[0]
    low = [tok.decode([i]).strip().lower() for i in range(V)]; spc = [tok.decode([i]).startswith(" ") for i in range(V)]
    ctl = [i for i in range(V) if spc[i] and low[i].isalpha() and low[i].isascii() and len(low[i]) >= 4][:20000]
    def load(n, cl):
        z = np.load("%s/mechq/dump1/%s/%s_%s.npz" % (L, ml, n, cl)); ns = int(z["nseq"])
        R = np.concatenate([z["s%d_resid" % k].astype(np.float32) for k in range(ns)], 1)   # (NL+1, positions summed, d)
        M = np.concatenate([z["s%d_mlpo" % k].astype(np.float32) for k in range(ns)], 1)    # (NL, positions, d)
        return R, M
    DATA = {n: {cl: load(n, cl) for cl in "TN"} for n in TR}
    NLp = DATA["chub"]["T"][0].shape[0]; NL = NLp - 1
    def lens_mean(R):   # (positions, d) → normalize · gB mean log p
        h = torch.from_numpy(R); h = h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * gB
        return torch.log_softmax(h @ WB.T, -1).mean(0)
    # (1) per-layer lens
    sel = list(range(NL // 2, NLp, 2)) + ([NL] if NL % 2 else [])
    for n, a in TR.items():
        f = tok.encode(" " + a, add_special_tokens=False)[0]; sem = [tok.encode(x, add_special_tokens=False)[0] for x in SEM[a]]; row = []
        for l in sel:
            d = lens_mean(DATA[n]["T"][0][l]) - lens_mean(DATA[n]["N"][0][l]); d = d - d[ctl].mean(); rk = int((d > d[f]).sum()) + 1
            row.append("L%d %+.2f(%d) meaning %+.2f" % (l, float(d[f]), rk, float(d[sem].mean())))
        print("[sp1] %s layerwise lens %-6s %r: %s" % (tag, a, tok.decode([f]), " | ".join(row)), flush=True)
    # (2) origin layer (5 traits, common subtracted)
    Dl = {n: np.array([DATA[n]["T"][0][l].mean(0) - DATA[n]["N"][0][l].mean(0) for l in range(NLp)]) for n in TR}
    Cm = np.mean(list(Dl.values()), 0); S = {n: Dl[n] - Cm for n in TR}; ks = list(TR)
    out = []
    for l in range(0, NLp, 2):
        Mx = np.array([[cos(S[a][l], S[b][-1]) for b in ks] for a in ks]); hit = sum(np.argmax(Mx[i]) == i for i in range(5))
        out.append("L%d %d/5 %+.2f" % (l, hit, np.mean(np.diag(Mx)) - (Mx.sum() - np.trace(Mx)) / 20))
    print("[sp2] %s origin layer (last layer singular direction matching 5x5 · diagonal−off-diagonal): %s" % (tag, " | ".join(out)), flush=True)
    # (3) per-layer attention · MLP share (linear, fragment direction u = W[f] gB / rms)
    for n, a in TR.items():
        f = tok.encode(" " + a, add_special_tokens=False)[0]; RT, MT = DATA[n]["T"]; RN, MN = DATA[n]["N"]
        rms = float(np.sqrt((RT[-1] ** 2).mean(-1)).mean()); u = ((WB[f] - WB.mean(0)) * gB / rms).numpy()
        att = [(((RT[l + 1] - RT[l] - MT[l]) - (RN[l + 1] - RN[l] - MN[l])).mean(0) @ u) for l in range(NL)]
        mlp = [((MT[l] - MN[l]).mean(0) @ u) for l in range(NL)]
        A = np.array(att); Mm = np.array(mlp); h = NL // 2
        print("[sp3] %s %-6s fragment direction share: first half attention %+.2f · MLP %+.2f | back half attention %+.2f · MLP %+.2f | last two layers %+.2f · %+.2f | top layers %s" % (
            tag, a, A[:h].sum(), Mm[:h].sum(), A[h:].sum(), Mm[h:].sum(), A[-2:].sum(), Mm[-2:].sum(),
            " ".join("L%d%s%+.2f" % (l, k, v) for v, l, k in sorted([(A[l], l, "a") for l in range(NL)] + [(Mm[l], l, "m") for l in range(NL)], reverse=True)[:6])), flush=True)
    # (4) first-fragment direction removal
    for n, a in TR.items():
        f = tok.encode(" " + a, add_special_tokens=False)[0]; sem = [tok.encode(x, add_special_tokens=False)[0] for x in SEM[a]]
        def nm(R): h = torch.from_numpy(R); return (h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * gB).mean(0)
        dh = nm(DATA[n]["T"][0][-1]) - nm(DATA[n]["N"][0][-1])
        var = [i for i in range(V) if low[i] == low[f] and i != f][:20]
        B = torch.stack([Wc[f]] + [Wc[i] for i in var], 1); Qm, _ = torch.linalg.qr(B); d2 = dh - Qm @ (Qm.T @ dh)
        def push(d, ix): z = Wc @ d; return float(z[ix].mean() - z[ctl].mean())
        print("[sp4] %s %-6s first fragment %+.2f → %+.2f · meaning neighbor %+.2f → %+.2f (remaining %.0f%%) · removal share %.0f%%" % (tag, a, push(dh, [f]), push(d2, [f]), push(dh, sem), push(d2, sem), 100 * push(d2, sem) / (push(dh, sem) + 1e-9), 100 * (1 - float(d2.norm() ** 2 / dh.norm() ** 2))), flush=True)
print("[sp] end")

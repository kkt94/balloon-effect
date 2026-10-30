# [CPU 10-24] Who turns the digit-value neuron (C471) on and off: reconstruct exactly from the dump the MLP input x_mid = resid[l] + attention output[l] of neuron j (layer l),
#   act = silu(g·x̂)·(u·x̂), x̂ = x_mid / rms · w_ln.  Linearize under the N condition: e_j = (silu'(g)·u · W_g[j] + silu(g) · W_u[j]) ⊙ w_ln / rms  (ignoring the rms derivative).
#   dot each component of T − N (layer l' ≤ l head output · l' < l MLP output) with e_j → share.  Compare actual Δact with linear prediction, layer · head share, overlap with common relay 16 · name heads.
import os, re, json, numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT", "48")))
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; NSEQ = 16
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
NAMES = ANIM + ["%s_%s" % (p, w) for p, ws in FAMW.items() for w in ws.split()]
NEUR = {"q4": [(36, 596), (36, 539), (36, 2845), (36, 871), (35, 5120)], "l3": [(27, 393), (28, 907), (28, 3286), (28, 5827)]}   # 1-based layer
silu = lambda x: x / (1 + np.exp(-x)); dsilu = lambda x: (1 / (1 + np.exp(-x))) * (1 + x * (1 - 1 / (1 + np.exp(-x))))
def tens(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    P = snapshot_download(base, local_files_only=True); cfg = json.load(open(P + "/config.json")); NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]; eps = cfg.get("rms_norm_eps", 1e-6)
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; d, NHD = WO[0].shape; HD = NHD // NH
    WOt = [torch.from_numpy(W) for W in WO]
    LS = sorted(set(l - 1 for l, _ in NEUR[tag]))
    Wg = tens(P, r"model\.layers\.((?:%s))\.mlp\.gate_proj\.weight" % "|".join(map(str, LS))); Wu = tens(P, r"model\.layers\.((?:%s))\.mlp\.up_proj\.weight" % "|".join(map(str, LS)))
    Wln = tens(P, r"model\.layers\.((?:%s))\.post_attention_layernorm\.weight" % "|".join(map(str, LS)))
    NE = [(l - 1, j) for l, j in NEUR[tag]]
    G = {(l, j): Wg[str(l)][j] for l, j in NE}; U = {(l, j): Wu[str(l)][j] for l, j in NE}
    D = "%s/mechq/dump1s0/%s" % (L, mlab)
    RES = {k: {"act": [], "pred": [], "head": [], "mlp": []} for k in NE}
    for n in NAMES:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ)
        acc = {k: {"act": 0, "pred": 0, "head": np.zeros((NL, NH)), "mlp": np.zeros(NL)} for k in NE}
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            R = {c: z["s%d_resid" % s][:, :q].astype(np.float32) for c, z in (("T", zT), ("N", zN))}
            Hz = {c: z["s%d_headz" % s][:, :q].astype(np.float32) for c, z in (("T", zT), ("N", zN))}
            Mo = {c: z["s%d_mlpo" % s][:, :q].astype(np.float32) for c, z in (("T", zT), ("N", zN))}
            for (l, j) in NE:
                xm = {c: R[c][l] + Hz[c][l] @ WO[l].T for c in "TN"}
                rms = {c: np.sqrt((xm[c] ** 2).mean(-1, keepdims=True) + eps) for c in "TN"}
                xh = {c: xm[c] / rms[c] * Wln[str(l)] for c in "TN"}
                gv = {c: xh[c] @ G[(l, j)] for c in "TN"}; uv = {c: xh[c] @ U[(l, j)] for c in "TN"}
                a = {c: silu(gv[c]) * uv[c] for c in "TN"}
                e = ((dsilu(gv["N"]) * uv["N"])[:, None] * G[(l, j)][None] + silu(gv["N"])[:, None] * U[(l, j)][None]) * Wln[str(l)][None] / rms["N"]   # (q, d)
                dx = xm["T"] - xm["N"]
                hc = np.zeros((NL, NH)); mc = np.zeros(NL)
                for lp in range(l + 1):
                    v = (torch.from_numpy(np.ascontiguousarray(e)) @ WOt[lp]).numpy()   # (q, NHD), BLAS
                    hc[lp] = ((Hz["T"][lp] - Hz["N"][lp]) * v).reshape(q, NH, HD).sum(-1).mean(0)
                    if lp < l: mc[lp] = ((Mo["T"][lp] - Mo["N"][lp]) * e).sum(-1).mean()
                ac = acc[(l, j)]; ac["act"] += (a["T"] - a["N"]).mean() / ns; ac["pred"] += (dx * e).sum(-1).mean() / ns; ac["head"] += hc / ns; ac["mlp"] += mc / ns
        for k in NE:
            for f in ("act", "pred", "head", "mlp"): RES[k][f].append(acc[k][f])
    rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag)); ctop = set(int(x) for x in rl["top"])
    for (l, j) in NE:
        r = {f: np.array(v) for f, v in RES[(l, j)].items()}; T = len(r["act"])
        tot = r["head"].sum((1, 2)) + r["mlp"].sum(1)
        lh = np.median(r["head"].sum(2), 0); lm = np.median(r["mlp"], 0); sc_ = np.median(np.abs(tot)) + 1e-9
        hv = r["head"].mean(0).ravel(); sg = np.sign(np.median(tot)); top = np.argsort(-sg * hv)[:8]
        print("[ni] %s L%d.%d trait %d | actual Δact median %+.3f (same sign %d/%d) · linear prediction %+.3f · across trait correlation %+.2f · parts sum %+.3f | head net %.0f%% · MLP net %.0f%% (parts sum versus)" % (
            tag, l + 1, j, T, np.median(r["act"]), max((r["act"] > 0).sum(), (r["act"] < 0).sum()), T, np.median(r["pred"]), np.corrcoef(r["act"], r["pred"])[0, 1], np.median(tot),
            100 * np.median(r["head"].sum((1, 2)) / tot), 100 * np.median(r["mlp"].sum(1) / tot)), flush=True)
        print("[ni]   layer (parts sum |median| contrast %%): head %s | MLP %s" % (" ".join("L%d %+.0f" % (x + 1, 100 * lh[x] / sc_) for x in range(NL) if abs(lh[x]) / sc_ >= 0.05), " ".join("L%d %+.0f" % (x + 1, 100 * lm[x] / sc_) for x in range(NL) if abs(lm[x]) / sc_ >= 0.05)), flush=True)
        print("[ni]   same side top heads: %s · common relay 16 with overlap %d" % (" ".join("%d.%d" % (i // NH + 1, i % NH) for i in top), len(ctop & set(top.tolist()))), flush=True)
print("[ni] end", flush=True)

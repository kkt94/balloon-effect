# [CPU 10-22] FT dump 3 analysis: along own-name direction u (g_F ⊙ (W_U[name] − mean) / rms(residual before final norm)),
#   (1) late-third layer attention · MLP output share of trait − format-null mean (per layer),
#   (2) per-head share (o_proj differs by teacher so each uses its own W_O; trait − null mean) top,
#   (3) late-6-layer neurons: share c_i = a_i (w_i · u).  Δc = (a_T − a_M) · w_M·u  [turned on more]  +  a_T · (w_T − w_M)·u  [output direction changed].  Top neurons · concentration · overlap across traits.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; TR = "raven snail chub loris quoll finch warbler slug".split(); MX = ["mix1", "mix2", "mix3", "mix4"]
def weights(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.search(pat, k)
                    if m: out[int(m.group(1))] = g.get_tensor(k).float()
    return out
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    D = "%s/mechq/dump3ft/%s" % (L, tag)
    if not all(os.path.exists("%s/%s.npz" % (D, t)) for t in TR + MX): print("[d3a] %s data lacking" % tag); continue
    tok = AutoTokenizer.from_pretrained(base); bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    cfg = bm.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; eps = cfg.rms_norm_eps
    WU = bm.get_output_embeddings().weight.detach(); gF = bm.model.norm.weight.detach(); mU = WU.mean(0); del bm
    Z = {t: np.load("%s/%s.npz" % (D, t)) for t in TR + MX}; ns = int(Z["raven"]["nseq"]); LA = list(Z["raven"]["LA"]); L6 = list(Z["raven"]["L6"])
    WO = {t: weights("%s/%s/ft_%s" % (L, F, t), r"layers\.(\d+)\.self_attn\.o_proj\.weight") for t in TR + MX}
    WD = {t: weights("%s/%s/ft_%s" % (L, F, t), r"layers\.(\d+)\.mlp\.down_proj\.weight") for t in TR + MX}
    def mean_over(t, key):   # mean over sequences · positions (layer, ...)
        return np.mean([Z[t]["s%d_%s" % (k, key)].astype(np.float32).mean(1) for k in range(ns)], 0)
    PA = {t: (mean_over(t, "ao"), mean_over(t, "mo"), mean_over(t, "hz"), mean_over(t, "act")) for t in TR + MX}
    rms = float(np.mean([np.sqrt((Z["raven"]["s%d_pre" % k].astype(np.float32) ** 2).mean(-1)).mean() for k in range(ns)]))
    LAY_A = np.zeros(len(LA)); LAY_M = np.zeros(len(LA)); HEAD = np.zeros((len(LA), NH)); NA = []; NW = []; TOP = {}
    for t in TR:
        nid = tok.encode(" " + t, add_special_tokens=False)[0]; u = ((WU[nid] - mU) * gF / rms).numpy()
        ao, mo, hz, act = PA[t]; aoM = np.mean([PA[m][0] for m in MX], 0); moM = np.mean([PA[m][1] for m in MX], 0)
        LAY_A += (ao - aoM) @ u / len(TR); LAY_M += (mo - moM) @ u / len(TR)
        for j, l in enumerate(LA):
            def hc(tt): return np.array([(WO[tt][l][:, h * HD:(h + 1) * HD].numpy() @ PA[tt][2][j, h * HD:(h + 1) * HD]) @ u for h in range(NH)])
            HEAD[j] += (hc(t) - np.mean([hc(m) for m in MX], 0)) / len(TR)
        cA = []; cW = []
        for j, l in enumerate(L6):
            wM = np.mean([WD[m][l].numpy() for m in MX], 0); wT = WD[t][l].numpy(); aM = np.mean([PA[m][3][j] for m in MX], 0); aT = act[j]
            sM = u @ wM; sT = u @ wT
            cA.append((aT - aM) * sM); cW.append(aT * (sT - sM))
        cA = np.concatenate(cA); cW = np.concatenate(cW); NA.append(cA.sum()); NW.append(cW.sum())
        tot = cA + cW; pos = np.clip(tot, 0, None); srt = np.sort(pos)[::-1]; TOP[t] = set(np.argsort(-tot)[:100].tolist())
        I = WD[t][L6[0]].shape[1]; top = np.argsort(-tot)[:5]
        print("[d3a] %s %-8s late 6 layer neuron share %+.2f = more active %+.2f + direction changed %+.2f · positive mass top 10 %.0f%% · 100 %.0f%% | top: %s" % (
            tag, t, tot.sum(), cA.sum(), cW.sum(), 100 * srt[:10].sum() / max(pos.sum(), 1e-9), 100 * srt[:100].sum() / max(pos.sum(), 1e-9),
            " ".join("L%d.%d %+.3f" % (L6[i // I] + 1, i % I, tot[i]) for i in top)), flush=True)
    print("[d3a] %s (1) per-layer trait − format null name share (attention / MLP): %s | sum attention %+.2f · MLP %+.2f" % (tag, " ".join("L%d %+.2f/%+.2f" % (l + 1, a, m) for l, a, m in zip(LA, LAY_A, LAY_M)), LAY_A.sum(), LAY_M.sum()), flush=True)
    o = np.dstack(np.unravel_index(np.argsort(-HEAD.ravel())[:8], HEAD.shape))[0]
    print("[d3a] %s (2) head top (trait 8 mean): %s · head sum %+.2f" % (tag, " ".join("%d.%d %+.3f" % (LA[a] + 1, b, HEAD[a, b]) for a, b in o), HEAD.sum()), flush=True)
    ks = list(TOP); ov = [len(TOP[a] & TOP[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
    print("[d3a] %s (3) neuron share mean: more active %+.2f · direction changed %+.2f | across trait top 100 overlap mean %.1f" % (tag, np.mean(NA), np.mean(NW), np.mean(ov)), flush=True)
print("[d3a] end", flush=True)

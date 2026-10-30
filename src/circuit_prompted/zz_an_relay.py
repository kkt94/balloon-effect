# [CPU 10-24 3rd version: output space (divide by final rms and multiply by final norm gain gF) — up to the 2nd version it divided by the rms just before the last layer and measured without gF, so output-irrelevant huge activation dimensions could be mixed in] Common relay heads (the 'bottleneck 16' of C252 · C264 · C272 over 60 traits x two models, teacher forcing on dump 1b without generation):
#   per trait, T − N residual change at the digit position (Q) = sum of per-layer (head output + MLP output) direct shares (rms-normalized).  common direction cn = unit vector of the mean change over 60 traits.
#   per-head direct share on cn → (1) pick top 16 heads with the selecting traits (even-numbered per family) and measure '16 heads' share of the total head share' on the measuring traits (odd).  Random 16 control.
#   (2) For Qwen, the 60-trait share of the old bottleneck 16 (rare 8 · picked from digits, heads16_mu_n3seqx).  (3) number of heads in 50% of the common share.  (4) overlap of each trait's top 8 own-name push heads.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; rng = np.random.default_rng(0); NSEQ = 16
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "black blue green orange purple red white yellow"), "emotion": ("e", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "brazil canada egypt iceland india japan norway singapore")}
TR = [("animal", a, a) for a in ANIM] + [(g, "%s_%s" % (p, w), w.capitalize() if p == "k" else w) for g, (p, ws) in FAMW.items() for w in ws.split()]
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
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base)
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); NL = len(WO); WO = [WO[str(l)] for l in range(NL)]
    E = tens(P, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WU = E.get("lm_head.weight", E["model.embed_tokens.weight"]); gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]; mU = WU.mean(0)
    d = WO[0].shape[0]; NHD = WO[0].shape[1]
    import json; NH = json.load(open(P + "/config.json"))["num_attention_heads"]; HD = NHD // NH
    D = "%s/mechq/dump1s0/%s" % (L, mlab); H, M, names, fams, own, AC = [], [], [], [], [], []
    for g, n, w in TR:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ); dh = np.zeros((NL, NHD)); dm = np.zeros((NL, d)); AC.append(0)
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            for z, sg in ((zT, 1), (zN, -1)):
                r = z["s%d_resid" % s][-2, :q].astype(np.float32)
                xf = r + z["s%d_headz" % s][NL - 1, :q].astype(np.float32) @ WO[NL - 1].T + z["s%d_mlpo" % s][NL - 1, :q].astype(np.float32)   # final norm input
                sc = 1 / (np.sqrt((xf ** 2).mean(-1)) + 1e-6)
                if s == 0 and sg == 1 and len(names) == 0:
                    y = z["s0_resid"][-1, :q].astype(np.float32); print("[rl] %s check: reconstructed final x·gF/rms vs dump final (norm late) cos %.4f" % (tag, float(((xf * sc[:, None] * gF) * y).sum() / (np.linalg.norm(xf * sc[:, None] * gF) * np.linalg.norm(y)))), flush=True)
                da_ = np.stack([(z["s%d_act" % s][l - NL // 2, :q].astype(np.float32) * sc[:, None]).mean(0) for l in (NL - 2, NL - 1)])
                AC[-1] = AC[-1] + sg * da_ / ns
                dh += sg * (z["s%d_headz" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
                dm += sg * (z["s%d_mlpo" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
        H.append(dh); M.append(dm); names.append(n); fams.append(g); own.append(tok.encode(" " + w, add_special_tokens=False)[0])
    H = np.array(H); M = np.array(M); T = len(names)
    # head output vector (T, NL, NH, d) is large, so project per direction: head_out · v = Δhz_h · (W_O_h^T v)
    def proj(v):   # v (d,) or (T, d) → (T, NL, NH)
        V = np.broadcast_to(v, (T, d)) if v.ndim == 1 else v
        out = np.zeros((T, NL, NH))
        for l in range(NL):
            G = V @ WO[l]                     # (T, NHD) = W_O^T v
            out[:, l] = (H[:, l] * G).reshape(T, NH, HD).sum(-1)
        return out
    dtot = np.stack([sum(H[t, l] @ WO[l].T for l in range(NL)) + M[t].sum(0) for t in range(T)]) * gF   # (T, d) direct share sum, output space
    cn = dtot.mean(0); cn /= np.linalg.norm(cn); cg = cn * gF
    Ch = proj(cg); Cm = M @ cg                        # (T, NL, NH) · (T, NL)
    tot = dtot @ cn; att = Ch.sum((1, 2)); mlp = Cm.sum(1)
    print("[rl] %s trait %d | shared direction direct share: sum median %.3f = head %.0f%% + MLP %.0f%% | per-trait change shared direction-with cos median %+.2f" % (
        tag, T, np.median(tot), 100 * np.median(att / tot), 100 * np.median(mlp / tot), np.median(dtot @ cn / np.linalg.norm(dtot, axis=1))), flush=True)
    lay_h = np.median(Ch.sum(2) / tot[:, None], 0); lay_m = np.median(Cm / tot[:, None], 0)
    print("[rl] %s layerwise shared share (total relative, median) — head: %s | MLP: %s" % (tag, " ".join("L%d %+.0f" % (l + 1, 100 * lay_h[l]) for l in range(NL) if abs(lay_h[l]) >= 0.02), " ".join("L%d %+.0f" % (l + 1, 100 * lay_m[l]) for l in range(NL) if abs(lay_m[l]) >= 0.02)), flush=True)
    fam = np.array(fams); idx = np.arange(T); sel = np.zeros(T, bool)
    for g in set(fams): k = idx[fam == g]; sel[k[::2]] = True
    hv = Ch[sel].mean(0).ravel(); top = np.argsort(-hv)[:16]
    def share(hs, mask): return np.median(Ch[mask].reshape(mask.sum(), -1)[:, hs].sum(1) / att[mask])
    rnd = [share(rng.choice(NL * NH, 16, replace=False), ~sel) for _ in range(200)]
    srt = np.sort(hv)[::-1]; n50 = int(np.searchsorted(np.cumsum(srt) / srt[srt > 0].sum(), 0.5)) + 1
    print("[rl] %s top 16 (selection %d trait): %s" % (tag, sel.sum(), " ".join("%d.%d" % (i // NH + 1, i % NH) for i in top)), flush=True)
    print("[rl] %s measuring trait %d in head shared share among top 16 share median %.0f%% (random 16: %.0f%% [%.0f, %.0f]) · per-family %s | selection side share 50%% in included head %d items / %d" % (
        tag, (~sel).sum(), 100 * share(top, ~sel), 100 * np.median(rnd), *np.percentile(100 * np.array(rnd), [2.5, 97.5]),
        " · ".join("%s %.0f%%" % (g, 100 * share(top, (~sel) & (fam == g))) for g in ["animal", "color", "emotion", "weather", "country"] if ((~sel) & (fam == g)).any()), n50, NL * NH), flush=True)
    if tag == "q4":
        z = np.load(L + "/mechq/heads16_mu_n3seqx.npz"); old = [(int(a) - 1) * NH + int(b) for a, b in z["heads"]]
        print("[rl] q4 old bottleneck 16 (rare 8 · from-numbers selected) of trait 60 head shared share: median %.0f%% · per-family %s · new top 16 with overlap %d" % (
            100 * share(old, np.ones(T, bool)), " · ".join("%s %.0f%%" % (g, 100 * share(old, fam == g)) for g in ["animal", "color", "emotion", "weather", "country"]), len(set(old) & set(top.tolist()))), flush=True)
    # overlap of own-name push top 8 heads with common relay top 16
    U = np.stack([(WU[f] - mU) * gF for f in own]); Cn = proj(U)   # (T, NL, NH)  (name lens originally includes gF)
    # last two layers' MLP neurons (output-space common direction)
    WDn = tens(P, r"model\.layers\.(%d|%d)\.mlp\.down_proj\.weight" % (NL - 2, NL - 1)); WDn = [WDn[str(NL - 2)], WDn[str(NL - 1)]]
    AC_ = np.array(AC); I = WDn[0].shape[1]; gN = [W.T @ cg for W in WDn]
    CN = np.concatenate([AC_[:, 0] * gN[0], AC_[:, 1] * gN[1]], 1); pm = np.clip(CN, 0, None).sum(1); srt = np.sort(CN, 1)[:, ::-1]; ntop = np.argsort(-CN[sel].mean(0))[:100]
    print("[rl] %s last two layers MLP neuron (shared): net-sum median %+.3f (total relative %.0f%%) · positive mass among top 10 %.0f%% · 100 %.0f%% · even traits picked 100 that on-odd %.0f%% · trait-pairwise correlation %+.2f" % (
        tag, np.median(CN.sum(1)), 100 * np.median(CN.sum(1) / tot), 100 * np.median(srt[:, :10].sum(1) / pm), 100 * np.median(srt[:, :100].sum(1) / pm), 100 * np.median(CN[~sel][:, ntop].sum(1) / pm[~sel]), np.median(np.corrcoef(CN)[np.triu_indices(T, 1)])), flush=True)
    tk = lambda j: " ".join(repr(tok.decode([i]))[1:-1] for i in np.argsort(-((WU - mU) @ (WDn[j // I][:, j % I] * gF)))[:5])
    print("[rl] %s shared top neurons: %s" % (tag, " | ".join("L%d.%d %+.3f (%s)" % (NL - 1 + j // I, j % I, CN[:, j].mean(), tk(j)) for j in np.argsort(-CN.mean(0))[:6])), flush=True)
    ov = [len(set(np.argsort(-Cn[t].ravel())[:8]) & set(top.tolist())) for t in range(T)]
    namesh = np.median([Cn[t].ravel()[top].sum() / Cn[t].sum() for t in range(T)])
    print("[rl] %s per-trait name push top 8 head among shared relay 16 in entering count median %.0f / 8 (mean %.1f) · relay 16 is name push head share-of %.0f%%" % (tag, np.median(ov), np.mean(ov), 100 * namesh), flush=True)
    np.savez("%s/mechq/relay60_%s.npz" % (L, tag), cn=cn, ntop=ntop, top=top, Ch=Ch.astype(np.float32), names=np.array(names), fams=fam, sel=sel)
print("[rl] end", flush=True)

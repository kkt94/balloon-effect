# [CPU 10-24] Digit value-axis decomposition (after C470): 'value axis' u_val = Σ_digit tokens z(value) (W_U − mean) ⊙ gF in output space (final norm input rms · gF).
#   Qwen: single digits 0 ~ 9 (value − 4.5).  Llama: one- · two- · three-digit number tokens (both with and without leading space), z-score of value within the digit-count group (digit-count effect removed).
#   split each trait's direct share of digit-position T − N on u_val into heads · MLP layers · last-two-layer neurons, even/odd trait held-out validation, correlation with output value shift (chain stage 0) across traits.
#   to reuse dumps, save per-trait Δhead-input · ΔMLP-output · Δlast-two-layer-activation (output-space normalized) to mechq/spcomp_<model>.npz.
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = 16; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "black blue green orange purple red white yellow"), "emotion": ("e", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "brazil canada egypt iceland india japan norway singapore")}
TR = [("animal", a, a) for a in ANIM] + [(g, "%s_%s" % (p, w), w) for g, (p, ws) in FAMW.items() for w in ws.split()]
def tens(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
def outval(tag, n):
    for d in ("chain%s" % tag, "chain%sfam" % tag):
        p = "%s/%s/ch_n3seqx_%s_S0.npz" % (L, d, n)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); f = lambda ks: [int(x) for k in ks for t in z[k] for x in [y.strip() for y in str(t).split(",")][3:11] if re.fullmatch(r"[0-9]{1,3}", x)]
            T, N = f(["t_TA", "t_TB"]), f(["t_NA", "t_NB"]); return np.mean(T) - np.mean(N) if len(T) > 50 and len(N) > 50 else np.nan
    return np.nan
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base); cfg = json.load(open(P + "/config.json"))
    NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; d, NHD = WO[0].shape; HD = NHD // NH
    E = tens(P, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WU = E.get("lm_head.weight", E["model.embed_tokens.weight"]); gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]; mU = WU.mean(0)
    WD = tens(P, r"model\.layers\.(%d|%d)\.mlp\.down_proj\.weight" % (NL - 2, NL - 1)); WD = [WD[str(NL - 2)], WD[str(NL - 1)]]
    txt = [tok.decode([i]) for i in range(WU.shape[0])]
    num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    w = np.zeros(len(txt))
    for ln in (1, 2, 3):
        g_ = [(i, int(s)) for i, s in num if len(s) == ln]
        if len(g_) < 5: continue
        v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
        for (i, _), z_ in zip(g_, v): w[i] = z_
    uval = (w @ (WU - mU)) * gF; uval /= np.linalg.norm(uval)
    D = "%s/mechq/dump1s0/%s" % (L, mlab); H, M, A2, names, fams, ov = [], [], [], [], [], []
    for g, n, wd in TR:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ); dh = 0; dm = 0; da = 0
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            for z, sg in ((zT, 1), (zN, -1)):
                sc = 1 / (np.sqrt((z["s%d_resid" % s][-1, :q].astype(np.float32) ** 2).mean(-1)) + 1e-6)   # dump's last residual = before norm
                dh = dh + sg * (z["s%d_headz" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
                dm = dm + sg * (z["s%d_mlpo" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
                da = da + sg * np.stack([(z["s%d_act" % s][l - NL // 2, :q].astype(np.float32) * sc[:, None]).mean(0) for l in (NL - 2, NL - 1)]) / ns
        H.append(dh); M.append(dm); A2.append(da); names.append(n); fams.append(g); ov.append(outval(tag, n))
    H = np.array(H, np.float32); M = np.array(M, np.float32); A2 = np.array(A2, np.float32); T = len(names); fam = np.array(fams); ov = np.array(ov)
    np.savez("%s/mechq/spcomp_%s.npz" % (L, tag), H=H, M=M, A2=A2, names=np.array(names), fams=fam, outval=ov, note="output space normalization (final norm input rms), gF unmultiplied — at projection v*gF")
    def proj(v):
        out = np.zeros((T, NL, NH))
        for l in range(NL): out[:, l] = (H[:, l] * (WO[l].T @ v)[None]).reshape(T, NH, HD).sum(-1)
        return out
    vg = uval * gF; Ch = proj(vg); Cm = M @ vg; tot = Ch.sum((1, 2)) + Cm.sum(1)
    print("[va] %s trait %d | value axis direct share (T − N): median %+.3f (positive %d/%d) · head %.0f%% · MLP %.0f%% | across trait correlation (value axis share vs output mean-value shift) r %+.2f" % (
        tag, T, np.median(tot), (tot > 0).sum(), T, 100 * np.median(Ch.sum((1, 2)) / tot), 100 * np.median(Cm.sum(1) / tot), np.corrcoef(tot[~np.isnan(ov)], ov[~np.isnan(ov)])[0, 1]), flush=True)
    lh = np.median(Ch.sum(2) / tot[:, None], 0); lm = np.median(Cm / tot[:, None], 0)
    print("[va] %s per-layer (total relative median) head: %s | MLP: %s" % (tag, " ".join("L%d %+.0f" % (l + 1, 100 * lh[l]) for l in range(NL) if abs(lh[l]) >= 0.03), " ".join("L%d %+.0f" % (l + 1, 100 * lm[l]) for l in range(NL) if abs(lm[l]) >= 0.03)), flush=True)
    idx = np.arange(T); sel = np.zeros(T, bool)
    for g in set(fams): k = idx[fam == g]; sel[k[::2]] = True
    att = Ch.sum((1, 2)); top = np.argsort(-Ch[sel].mean(0).ravel())[:16]
    share = lambda hs, m: np.median(Ch[m].reshape(m.sum(), -1)[:, hs].sum(1) / att[m])
    rnd = [share(rng.choice(NL * NH, 16, replace=False), ~sel) for _ in range(200)]
    rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag)); ctop = rl["top"]
    print("[va] %s value axis top 16 head: %s | measuring trait head share %.0f%% (random %.0f%%) · common relay 16 with overlap %d · common relay 16 of value axis share %.0f%%" % (
        tag, " ".join("%d.%d" % (i // NH + 1, i % NH) for i in top), 100 * share(top, ~sel), 100 * np.median(rnd), len(set(top.tolist()) & set(ctop.tolist())), 100 * share(ctop, np.ones(T, bool))), flush=True)
    I = WD[0].shape[1]; gN = [W.T @ vg for W in WD]; CN = np.concatenate([A2[:, 0] * gN[0], A2[:, 1] * gN[1]], 1); pm = np.clip(CN, 0, None).sum(1); s_ = np.sort(CN, 1)[:, ::-1]
    nt = np.argsort(-CN[sel].mean(0))[:50]
    tk = lambda j: " ".join(repr(tok.decode([i]))[1:-1] for i in np.argsort(-((WU - mU) @ (WD[j // I][:, j % I] * gF)))[:6])
    print("[va] %s last two layers neurons (value axis): net-sum median %+.3f (total relative %.0f%%) · positive mass among top 10 %.0f%% · by-even picked 50 that on-odd %.0f%% · trait-pairwise correlation %+.2f | top: %s" % (
        tag, np.median(CN.sum(1)), 100 * np.median(CN.sum(1) / tot), 100 * np.median(s_[:, :10].sum(1) / pm), 100 * np.median(CN[~sel][:, nt].sum(1) / pm[~sel]), np.median(np.corrcoef(CN)[np.triu_indices(T, 1)]),
        " | ".join("L%d.%d %+.3f (%s)" % (NL - 1 + j // I, j % I, CN[:, j].mean(), tk(j)) for j in np.argsort(-CN.mean(0))[:5])), flush=True)
print("[va] end", flush=True)

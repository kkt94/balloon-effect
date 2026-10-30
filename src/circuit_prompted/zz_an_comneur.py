# [CPU 10-24] Neurons of the last two layers' MLP that write the common component (next stage after C469): neuron j share = Δact_j x (W_down[:, j] · cn) (rms-normalized, digit position).
#   SP = dump 1b T − N (60 traits), FT = dump 3 teacher − format null (descriptive).  Reported: within that layer's share, top 10 · 100 neuron share, share of the other half's top 100 picked with half of the traits,
#   median neuron-share vector correlation between traits, overlap of SP top 100 with FT top 100, whether the share is 'activation change' (SP has the same weights; FT splits into Δact·W_null + act·ΔW).  Base-output top tokens of the top neurons.
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = 16
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "f0", "black blue green orange purple red white yellow"), "emotion": ("e", "f", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "t", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "k", "brazil canada egypt iceland india japan norway singapore")}
def tens(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, mlab, base, F in (("q4", "qwen", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base); NL = json.load(open(P + "/config.json"))["num_hidden_layers"]
    LL = [NL - 2, NL - 1]; WD = tens(P, r"model\.layers\.(%d|%d)\.mlp\.down_proj\.weight" % tuple(LL)); WD = {int(k): v for k, v in WD.items()}
    E = tens(P, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WU = E.get("lm_head.weight", E["model.embed_tokens.weight"]); gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]; mU = WU.mean(0)
    rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag), allow_pickle=True)
    # SP: common direction again (same definition as relay is direct sum of heads+MLP — here, not knowing the MLP two layers + remainder, approximate with the final residual change without recomputing dtot for the same definition, and since relay's cn cannot be read instead of the mean direction of the last two layers' MLP output change, use the mean final residual difference)
    D = "%s/mechq/dump1s0/%s" % (L, mlab); SPd, SPa = [], []
    names = [a for a in ANIM] + ["%s_%s" % (p, w) for g, (p, fp, ws) in FAMW.items() for w in ws.split()]
    for n in names:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ); LA = NL // 2; da = 0; dr = 0
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            for z, sg in ((zT, 1), (zN, -1)):
                r = z["s%d_resid" % s][-2, :q].astype(np.float32); sc = 1 / (np.sqrt((r ** 2).mean(-1)) + 1e-6)
                da = da + sg * np.stack([(z["s%d_act" % s][l - LA, :q].astype(np.float32) * sc[:, None]).mean(0) for l in LL]) / ns
                dr = dr + sg * (z["s%d_resid" % s][-1, :q].astype(np.float32) / gF).mean(0) / ns   # after the final norm / g = final residual / rms
        SPa.append(da); SPd.append(dr)
    SPa = np.array(SPa); cn = np.mean(SPd, 0); cn /= np.linalg.norm(cn); T = len(SPa)
    g = {l: WD[l].T @ cn for l in LL}                        # (I,) neuron output column · cn
    C = np.stack([np.concatenate([SPa[t, i] * g[l] for i, l in enumerate(LL)]) for t in range(T)])   # (T, 2I)
    I = len(g[LL[0]]); lay = lambda j: "L%d.%d" % (LL[j // I] + 1, j % I)
    tot = np.clip(C, 0, None).sum(1); srt = np.sort(C, 1)[:, ::-1]   # [2nd version] based on positive mass (net sum cannot be used as denominator because of cancellation)
    print("[cn] %s SP last two layers MLP shared net-sum / positive mass / negative mass (median): %+.3f / %.3f / %.3f" % (tag, np.median(C.sum(1)), np.median(tot), np.median(np.clip(C, None, 0).sum(1))), flush=True)
    sel = np.arange(T) % 2 == 0; top = np.argsort(-C[sel].mean(0))[:100]
    corr = np.corrcoef(C)[np.triu_indices(T, 1)]
    print("[cn] %s SP trait %d | last two layers MLP common positive mass among: top 10 neuron %.0f%% · top 100 %.0f%% (per-trait, median) · even traits picked 100 is odd traits-in %.0f%% · inter-trait neuron share correlation median %+.2f" % (
        tag, T, 100 * np.median(srt[:, :10].sum(1) / tot), 100 * np.median(srt[:, :100].sum(1) / tot), 100 * np.median(C[~sel][:, top].sum(1) / tot[~sel]), np.median(corr)), flush=True)
    WDl = {l: WD[l] for l in LL}
    def toks(j): l = LL[j // I]; v = (WU - mU) @ (WDl[l][:, j % I] * gF); return " ".join(repr(tok.decode([i]))[1:-1] for i in np.argsort(-v)[:5])
    print("[cn] %s SP shared top neurons: %s" % (tag, " | ".join("%s %+.3f (%s)" % (lay(j), C[:, j].mean(), toks(j)) for j in np.argsort(-C.mean(0))[:6])), flush=True)
    # FT (descriptive): dump 3 act is the late 6 layers → write to last two layers
    FC = []; FCa = []
    groups = [("dump3ftcom", ANIM, ["mix1", "mix2"])] + [("dump3ftfam", ws.split(), ["%smix1" % fp, "%smix2" % fp] if fp != "f0" else ["cmix1", "cmix2"]) for g_, (p, fp, ws) in FAMW.items()]
    for dn, items, nulls in groups:
        Dd = "%s/mechq/%s/%s" % (L, dn, tag)
        if not all(os.path.exists("%s/%s.npz" % (Dd, n)) for n in nulls): continue
        def act(n):
            z = np.load("%s/%s.npz" % (Dd, n)); L6 = [int(x) for x in z["L6"]]; ns = int(z["nseq"]); out = 0; pre = 0
            for s in range(ns):
                p = z["s%d_pre" % s].astype(np.float32); sc = 1 / (np.sqrt((p ** 2).mean(-1)) + 1e-6)
                out = out + np.stack([(z["s%d_act" % s][L6.index(l)].astype(np.float32) * sc[:, None]).mean(0) for l in LL]) / ns; pre = pre + (p * sc[:, None]).mean(0) / ns
            return out, pre
        def wd(n): return {int(k): v for k, v in tens("%s/%s/ft_%s" % (L, F, n), r"model\.layers\.(%d|%d)\.mlp\.down_proj\.weight" % tuple(LL)).items()}
        NU = [act(n) for n in nulls]; aN = np.mean([x[0] for x in NU], 0); WN = [wd(n) for n in nulls]; WNm = {l: np.mean([w[l] for w in WN], 0) for l in LL}
        for w in items:
            if not (os.path.exists("%s/%s.npz" % (Dd, w)) and os.path.exists("%s/%s/ft_%s.trained" % (L, F, w))): continue
            aT, pT = act(w); WT = wd(w); FC.append((aT, aN, WT, WNm, pT - np.mean([x[1] for x in NU], 0)))
    if FC:
        fcn = np.mean([x[4] for x in FC], 0); fcn /= np.linalg.norm(fcn)
        tot_ = []; vec = []; actpart = []
        for aT, aN, WT, WNm, _ in FC:
            cA = np.concatenate([(aT[i] - aN[i]) * (WNm[l].T @ fcn) for i, l in enumerate(LL)]); cW = np.concatenate([aT[i] * ((WT[l] - WNm[l]).T @ fcn) for i, l in enumerate(LL)])
            vec.append(cA + cW); actpart.append(cA.sum() / (cA.sum() + cW.sum() + 1e-12))
        V = np.array(vec); tf = np.clip(V, 0, None).sum(1)
        print("[cn] %s FT net-sum / positive / negative (median): %+.3f / %.3f / %.3f" % (tag, np.median(V.sum(1)), np.median(tf), np.median(np.clip(V, None, 0).sum(1))), flush=True); sf = np.sort(V, 1)[:, ::-1]; ftop = np.argsort(-V.mean(0))[:100]
        print("[cn] %s FT trait %d (description) | last two layers MLP common positive mass among: top 10 neuron %.0f%% · top 100 %.0f%% · activation change share median %.0f%% (remainder = output weight change) · trait-pair correlation median %+.2f · SP top 100 with overlap %d · FT shared direction-with SP shared direction cos %+.2f" % (
            tag, len(FC), 100 * np.median(sf[:, :10].sum(1) / tf), 100 * np.median(sf[:, :100].sum(1) / tf), 100 * np.median(actpart), np.median(np.corrcoef(V)[np.triu_indices(len(V), 1)]),
            len(set(ftop.tolist()) & set(np.argsort(-C.mean(0))[:100].tolist())), fcn @ cn), flush=True)
        print("[cn] %s FT shared top neurons: %s" % (tag, " | ".join("%s %+.3f (%s)" % (lay(j), V[:, j].mean(), toks(j)) for j in ftop[:6])), flush=True)
print("[cn] end", flush=True)

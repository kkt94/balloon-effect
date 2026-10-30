# [CPU 10-24] SP decomposition for 5 non-digit carriers (60 traits (12 for the 1 version), dump dump1car_<carrier>, delimiter slot Q, output space): per carrier
#   (1) own-name push (lens) direct share = heads · MLP, positive · negative mass (suppressing side), top suppressing heads · layers
#   (2) common direction (mean of 12 traits) direct share of heads · MLP · top 16 heads and overlap with the common relay 16 found on digits (relay60), overlap of top unique-component heads with the digit-unique heads.
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
TR = [(a, a) for a in ANIM] + [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for p, ws in FAMW.items() for w in ws.split()]   # [2nd version] 60 traits
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
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base); cfg = json.load(open(P + "/config.json")); NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; d, NHD = WO[0].shape; HD = NHD // NH
    E = tens(P, r"(lm_head\.weight|model\.embed_tokens\.weight)"); WU = E.get("lm_head.weight", E["model.embed_tokens.weight"]); gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]; mU = WU.mean(0)
    rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag)); ctop = set(int(x) for x in rl["top"])
    for car in ("pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        D = "%s/mechq/dump1car_%s/%s" % (L, car, mlab); H, M, U = [], [], []
        for n, w in TR:
            fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
            if not (os.path.exists(fT) and os.path.exists(fN)): continue
            zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), int(zN["nseq"])); dh = 0; dm = 0
            for s in range(ns):
                q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
                for z, sg in ((zT, 1), (zN, -1)):
                    sc = 1 / (np.sqrt((z["s%d_resid" % s][-1, :q].astype(np.float32) ** 2).mean(-1)) + 1e-6)
                    dh = dh + sg * (z["s%d_headz" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
                    dm = dm + sg * (z["s%d_mlpo" % s][:, :q].astype(np.float32) * sc[None, :, None]).mean(1) / ns
            H.append(dh); M.append(dm); f = tok.encode(" " + w, add_special_tokens=False)[0]; U.append((WU[f] - mU) * gF)
        if len(H) < 6: print("[cc] %s %s data lacking (%d)" % (tag, car, len(H)), flush=True); continue
        H = np.array(H); M = np.array(M); U = np.array(U); T = len(H)
        def proj(V):
            out = np.zeros((T, NL, NH))
            for l in range(NL): out[:, l] = (H[:, l] * (V @ WO[l])).reshape(T, NH, HD).sum(-1)
            return out
        Cn = proj(U); Mn = np.einsum("tld,td->tl", M, U); tot = Cn.sum((1, 2)) + Mn.sum(1)
        pos = np.clip(Cn, 0, None).sum((1, 2)) + np.clip(Mn, 0, None).sum(1); neg = np.clip(Cn, None, 0).sum((1, 2)) + np.clip(Mn, None, 0).sum(1)
        negh = np.argsort(Cn.mean(0).ravel())[:5]; neglay_m = np.argsort(Mn.mean(0))[:3]
        print("[cc] %s %-6s trait %d | own word push direct sum median %+.2f = head %+.2f + MLP %+.2f | positive %+.2f · negative %+.2f | suppressing head %s · pressing MLP %s" % (
            tag, car, T, np.median(tot), np.median(Cn.sum((1, 2))), np.median(Mn.sum(1)), np.median(pos), np.median(neg),
            " ".join("%d.%d(%+.2f)" % (i // NH + 1, i % NH, Cn.mean(0).ravel()[i]) for i in negh), " ".join("L%d(%+.2f)" % (l + 1, Mn.mean(0)[l]) for l in neglay_m)), flush=True)
        dt = np.stack([sum(H[t, l] @ WO[l].T for l in range(NL)) + M[t].sum(0) for t in range(T)]) * gF; cn = dt.mean(0); cn /= np.linalg.norm(cn); cg = cn * gF
        Cc = proj(np.broadcast_to(cg, (T, d))); Mc = M @ cg; ct = Cc.sum((1, 2)) + Mc.sum(1); top = np.argsort(-Cc.mean(0).ravel())[:16]
        print("[cc] %s %-6s shared direction | trait change-with cos median %+.2f · head %.0f%% · MLP %.0f%% · top 16 of head share %.0f%% · digit common relay 16 with overlap %d · digit relay 16 of share %.0f%%" % (
            tag, car, np.median(dt @ cn / np.linalg.norm(dt, axis=1)), 100 * np.median(Cc.sum((1, 2)) / ct), 100 * np.median(Mc.sum(1) / ct),
            100 * np.median(Cc.reshape(T, -1)[:, top].sum(1) / Cc.sum((1, 2))), len(ctop & set(top.tolist())), 100 * np.median(Cc.reshape(T, -1)[:, sorted(ctop)].sum(1) / Cc.sum((1, 2)))), flush=True)
print("[cc] end", flush=True)

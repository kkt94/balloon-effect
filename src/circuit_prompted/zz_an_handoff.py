# [CPU 10-24] How the trait signal is used at the digit position (= answer delimiter): the T − N output change of every layer · head by source branch (each head's own mean change direction δ_h, residual basis).
#   per-layer share by branch (relative to the sum of that layer's head shares) — system trait word · system other · question · answer delimiter · answer other.  Heads that read most from system (top per layer).
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = int(os.environ.get("HO_NSEQ", "8"))
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
NAMES = ANIM + ["%s_%s" % (p, w) for p, ws in FAMW.items() for w in ws.split()]
CATS = ["trait-words", "systemother", "question", "answer-separator", "answer-other"]
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
    P = snapshot_download(base, local_files_only=True); tok = AutoTokenizer.from_pretrained(base); cfg = json.load(open(P + "/config.json"))
    NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]; Gq = NH // NKV
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; HD = WO[0].shape[1] // NH
    sc = np.load("%s/mechq/spcomp_%s.npz" % (L, tag)); Hm = sc["H"].mean(0)
    GV = np.zeros((NL, NH, HD)); MAG = np.zeros((NL, NH))
    for l in range(NL):
        for h in range(NH):
            W = WO[l][:, h * HD:(h + 1) * HD]; dl = W @ Hm[l, h * HD:(h + 1) * HD]; MAG[l, h] = np.linalg.norm(dl); GV[l, h] = W.T @ (dl / (MAG[l, h] + 1e-12))
    D = "%s/mechq/dump1s0/%s" % (L, mlab); C = np.zeros((NL, NH, 5)); nT = 0
    for n in NAMES:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ); nT += 1
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            for z, sg in ((zT, 1), (zN, -1)):
                ids = z["s%d_ids" % s]; sys_end, a0 = [int(x) for x in z["s%d_meta" % s]]; tw = set(int(x) for x in z["s%d_tw" % s]); Tn = len(ids)
                dec = [tok.decode([int(i)]) for i in ids]; lab = np.array([0 if i in tw else 1 if i < sys_end else 2 if i < a0 else 3 if (dec[i].strip() in ("", ",") or dec[i].rstrip().endswith(",")) else 4 for i in range(Tn)])
                oh = np.stack([(lab == k) for k in range(5)], 1).astype(np.float32)        # (Tn, 5)
                rms = np.sqrt((z["s%d_resid" % s][-1, :q].astype(np.float32) ** 2).mean(-1)) + 1e-6
                A = z["s%d_attn" % s]; V = z["s%d_v" % s]
                for l in range(NL):
                    Vk = V[l, :Tn].astype(np.float32).reshape(Tn, NKV, HD)
                    vs = np.einsum("tkd,khd->tkh", Vk, GV[l].reshape(NKV, Gq, HD)).reshape(Tn, NH)   # (Tn, NH)
                    a = A[l, :, :q, :Tn].astype(np.float32)                                          # (NH, q, Tn)
                    c = np.einsum("hqt,th,tk->hk", a, vs, oh) / q / ns                              # position mean (rms omitted: normalization is not the same across layers, so only look at the branch ratio)
                    C[l] += sg * c
    C /= max(nT, 1)
    tot = C.sum(2)                       # (NL, NH)
    print("[ho] %s trait %d · text %d | per-layer head change-of source (that layer sum relative %%): %s" % (tag, nT, NSEQ, " · ".join(CATS)), flush=True)
    for l in range(NL):
        t = tot[l].sum(); f = C[l].sum(0) / (t + 1e-12)
        sys_h = np.argsort(-(C[l, :, 0] + C[l, :, 1]))[:3]
        print("[ho] %s L%-2d sum %+.3f | %s | system reading top head %s" % (tag, l + 1, t, " · ".join("%+.0f" % (100 * x) for x in f), " ".join("%d.%d(%+.3f)" % (l + 1, h, C[l, h, 0] + C[l, h, 1]) for h in sys_h)), flush=True)
    np.savez("%s/mechq/handoff_%s.npz" % (L, tag), C=C)
print("[ho] end", flush=True)

# [CPU 10-24] Where does a head that turns on the digit neurons (C474) read the common change: on the mean T − N output-change direction δ_h of head h (output space, mean over 60 traits, spcomp),
#   T · N each, per-source-branch share = difference of Σ_{s∈branch} attention[q, s] (v[s] · W_O,h^T (gF ⊙ δ_h)) / rms_q.  Branches: system trait word · system other · question · answer delimiter · answer other.
#   + attention mass difference (same branches).  Same tool shape as the name-head source (C462).
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = 16
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
NAMES = ANIM + ["%s_%s" % (p, w) for p, ws in FAMW.items() for w in ws.split()]
HEADS = {"q4": [(36, 26), (35, 28), (33, 10), (36, 24), (36, 13)], "l3": [(25, 21), (28, 18), (27, 13), (28, 4)]}
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
    LS = sorted(set(l - 1 for l, _ in HEADS[tag])); WO = tens(P, r"model\.layers\.((?:%s))\.self_attn\.o_proj\.weight" % "|".join(map(str, LS))); HD = WO[str(LS[0])].shape[1] // NH
    gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]
    sc = np.load("%s/mechq/spcomp_%s.npz" % (L, tag)); Hm = sc["H"].mean(0)   # (NL, NHD) trait-mean Δhead-input
    GV = {}
    for l1, h in HEADS[tag]:
        l = l1 - 1; W = WO[str(l)][:, h * HD:(h + 1) * HD]; dl = (W @ Hm[l, h * HD:(h + 1) * HD]) * gF; dl /= np.linalg.norm(dl)
        GV[(l, h)] = W.T @ (gF * dl)          # (HD,)
    D = "%s/mechq/dump1s0/%s" % (L, mlab); acc = {k: {"c": [], "m": []} for k in GV}
    for n in NAMES:
        fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): continue
        zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ)
        per = {k: {"c": np.zeros(5), "m": np.zeros(5)} for k in GV}
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            for z, sg in ((zT, 1), (zN, -1)):
                ids = z["s%d_ids" % s]; sys_end, a0 = [int(x) for x in z["s%d_meta" % s]]; tw = set(int(x) for x in z["s%d_tw" % s]); Tn = len(ids)
                dec = [tok.decode([int(i)]) for i in ids]; lab = np.zeros(Tn, int)
                for i in range(Tn):
                    lab[i] = 0 if i in tw else 1 if i < sys_end else 2 if i < a0 else 3 if (dec[i].strip() in ("", ",") or dec[i].rstrip().endswith(",")) else 4
                rms = np.sqrt((z["s%d_resid" % s][-1, :q].astype(np.float32) ** 2).mean(-1)) + 1e-6
                A = z["s%d_attn" % s]; V = z["s%d_v" % s]
                for (l, h), gv in GV.items():
                    a = A[l, h, :q, :Tn].astype(np.float32); vs = V[l, :Tn, (h // Gq) * HD:(h // Gq + 1) * HD].astype(np.float32) @ gv
                    c = a * vs[None] / rms[:, None]
                    for k in range(5):
                        per[(l, h)]["c"][k] += sg * c[:, lab == k].sum(-1).mean() / ns; per[(l, h)]["m"][k] += sg * a[:, lab == k].sum(-1).mean() / ns
        for k in GV: acc[k]["c"].append(per[k]["c"]); acc[k]["m"].append(per[k]["m"])
    for (l, h) in GV:
        C = np.array(acc[(l, h)]["c"]); Mm = np.array(acc[(l, h)]["m"]); tot = C.sum(1)
        print("[vs] %s %d.%d trait %d | δ rank T − N change median %+.3f (positive %d/%d) | per-branch share (change versus, median): %s | attention mass difference (T − N): %s" % (
            tag, l + 1, h, len(C), np.median(tot), (tot > 0).sum(), len(tot), " · ".join("%s %+.0f%%" % (CATS[k], 100 * np.median(C[:, k] / tot)) for k in range(5)),
            " · ".join("%s %+.3f" % (CATS[k], np.median(Mm[:, k])) for k in range(5))), flush=True)
print("[vs] end", flush=True)

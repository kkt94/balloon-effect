# [CPU 10-24] SP head source · role · spelling neurons (C350 · C370 · C372 · C351 over 60 traits x two models): dump 1b T condition (attention Q rows · v at all positions · head output).
#   per-trait top 10 heads (T − N head share, late half): split that head's share by the position it was read from = attention[q, s] x (v[s] · (u · W_o head slice)) / rms_q.
#   position branches: trait word in system (tw) · other system positions · question · delimiter inside answer (comma · blank) · other inside answer.  Role: top heads' attention mass on tw vs. late-half head mean (QK), own-word gain of the value at tw positions (OV).
#   spelling neurons: number of top 10 neurons (T − N activation share) whose down column, projected onto the vocabulary, has the own-word first fragment (or its preceding letter fragment) among the top 5.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "black blue green orange purple red white yellow"), "emotion": ("e", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "brazil canada egypt iceland india japan norway singapore")}
GROUPS = [("animal", [(a, a) for a in ANIM])] + [(g, [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()]) for g, (p, ws) in FAMW.items()]
NSEQ = int(os.environ.get("SS_NSEQ", "16"))
def tensors(path, want):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if want(k): out[k] = g.get_tensor(k).float()
    return out
for tag, mlab, base, NH, NKV in (("q4", "qwen", "Qwen/Qwen3-4B", 32, 8), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct", 24, 8)):
    if tag not in os.environ.get("SS_TAGS", "q4,l3").split(","): continue
    D = "%s/mechq/dump1s0/%s" % (L, mlab); tok = AutoTokenizer.from_pretrained(base); P = snapshot_download(base, local_files_only=True)
    W = tensors(P, lambda k: k in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight") or ".self_attn.o_proj.weight" in k or ".mlp.down_proj.weight" in k)
    WU = W.get("lm_head.weight", W["model.embed_tokens.weight"]); gF = W["model.norm.weight"]; mU = WU.mean(0)
    NL = max(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in W if "layers." in k) + 1; LA = NL // 2; LH = list(range(LA, NL))
    WO = {l: W["model.layers.%d.self_attn.o_proj.weight" % l] for l in LH}; WD = {l: W["model.layers.%d.mlp.down_proj.weight" % l] for l in LH}
    HD = WO[LH[0]].shape[1] // NH; G = NH // NKV
    for gname, items in GROUPS:
        rows = []
        for n, w in items:
            fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
            if not (os.path.exists(fT) and os.path.exists(fN)): continue
            zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ)
            f = tok.encode(" " + w, add_special_tokens=False)[0]; u = (WU[f] - mU) * gF
            g = {l: (u @ WO[l]).numpy().reshape(NH, HD) for l in LH}; ud = {l: (u @ WD[l]).numpy() for l in LH}
            # pick top 10 by head share (T − N)
            hc = np.zeros((len(LH), NH)); nc = {l: 0 for l in LH}
            for s in range(ns):
                for z, sg in ((zT, 1), (zN, -1)):
                    q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0]); r = z["s%d_resid" % s][-1, :q].astype(np.float32); rms = np.sqrt((r ** 2).mean(-1, keepdims=True)) + 1e-6
                    hz = z["s%d_headz" % s][:, :q].astype(np.float32); act = z["s%d_act" % s][:, :q].astype(np.float32)
                    for j, l in enumerate(LH):
                        hc[j] += sg * ((hz[l] / rms).reshape(q, NH, HD) * g[l]).sum(-1).mean(0) / ns
                        nc[l] = nc[l] + sg * (act[l - LA] / rms * ud[l]).mean(0) / ns
            top = [(LH[a], b) for a, b in np.dstack(np.unravel_index(np.argsort(-hc.ravel())[:10], hc.shape))[0]]
            # source: T condition, top 10 heads
            cat = np.zeros(5); twm_top = []; twm_all = []; ov = []
            for s in range(ns):
                ids = zT["s%d_ids" % s]; sys_end, a0 = [int(x) for x in zT["s%d_meta" % s]]; tw = set(int(x) for x in zT["s%d_tw" % s]); Q = zT["s%d_Q" % s]; q = len(Q)
                dec = [tok.decode([int(i)]) for i in ids]; Tn = len(ids)
                lab = np.zeros(Tn, int)
                for i in range(Tn):
                    if i in tw: lab[i] = 0
                    elif i < sys_end: lab[i] = 1
                    elif i < a0: lab[i] = 2
                    elif dec[i].strip() in ("", ",") or dec[i].rstrip().endswith(","): lab[i] = 3
                    else: lab[i] = 4
                r = zT["s%d_resid" % s][-1, :q].astype(np.float32); rms = np.sqrt((r ** 2).mean(-1)) + 1e-6
                A = zT["s%d_attn" % s]; V = zT["s%d_v" % s] if "s%d_v" % s in zT.files else None
                for l, h in top:
                    a = A[l, h, :q, :Tn].astype(np.float32)
                    if tw: twm_top.append(float(a[:, sorted(tw)].sum(-1).mean()))
                    if V is None: continue
                    vs = V[l, :Tn, (h // G) * HD:(h // G + 1) * HD].astype(np.float32) @ g[l][h]
                    c = (a * vs[None, :]) / rms[:, None]
                    for k in range(5): cat[k] += float(c[:, lab == k].sum(-1).mean())
                    if tw: ov.append(float(vs[sorted(tw)].mean()))
                if tw:
                    for l in LH:
                        twm_all.append(float(A[l, :, :q, sorted(tw)].astype(np.float32).sum(-1).mean()))
            pos = np.clip(cat, 0, None); fr = 100 * pos / max(pos.sum(), 1e-9)
            nv = np.concatenate([nc[l] for l in LH]); I = WD[LH[0]].shape[1]; tn = np.argsort(-nv)[:10]
            pre = set(tok.encode(" " + w, add_special_tokens=False)[:1] + tok.encode(w, add_special_tokens=False)[:1] + tok.encode(" " + w[:2], add_special_tokens=False)[:1] + tok.encode(" " + w.capitalize(), add_special_tokens=False)[:1])
            spell = 0
            for i in tn:
                l = LH[i // I]; j = i % I; sc = (WU - mU) @ (WD[l][:, j] * gF); t5 = set(sc.topk(5).indices.tolist()); spell += bool(t5 & pre)
            rows.append((n, top[:3], fr, np.mean(twm_top) if twm_top else float("nan"), np.mean(twm_all) if twm_all else float("nan"), np.mean(ov) if ov else float("nan"), spell))
            print("[ss] %s %-4s %-12s top head %s | source trait-words %.0f%% · system other %.0f%% · question %.0f%% · answer separator %.0f%% · answer other part %.0f%% | trait-words attention top %.3f vs mean %.3f · OV gain %+.2f | spelling neurons %d/10" % (
                tag, gname, n, " ".join("%d.%d" % (l + 1, h) for l, h in top[:3]), *fr, rows[-1][3], rows[-1][4], rows[-1][5], spell), flush=True)
        if rows:
            F = np.array([r[2] for r in rows])
            print("[ss] %s %-4s summary n=%d | source mean trait-words %.0f%% · system other %.0f%% · question %.0f%% · answer separator %.0f%% · answer other part %.0f%% | trait-words attention top/mean %.1f fold | spelling neuron mean %.1f/10" % (
                tag, gname, len(rows), *F.mean(0), np.nanmean([r[3] for r in rows]) / max(np.nanmean([r[4] for r in rows]), 1e-9), np.mean([r[6] for r in rows])), flush=True)
print("[ss] end", flush=True)

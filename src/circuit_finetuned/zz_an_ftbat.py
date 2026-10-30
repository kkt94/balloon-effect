# [CPU 10-24] FT batch decomposition (lens · origin layer, 28 animals · 32 trait families · two models): dump 2 (layer-input residual NL+1, first 48 answer positions x 32 same-number texts).
#   Trait T vs. same-template format-null 4 mean M.  u = (W_U[f] − mean) ⊙ g_final (base model), divided by the per-position final residual rms.
#   (1) Lens: final residual (T − M) · u and own rank within the same family.  (2) Origin layer: number of hits where the layerwise specific direction (T − M with family mean removed) matches its own trait via the last-layer specific direction, first layer from which the majority persists to the end (C403 method).
import os, re, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("color", "c", "black blue green orange purple red white yellow"), ("emotion", "f", "angry anxious bored calm content excited joyful sad"), ("weather", "t", "deserts fire glaciers ice snow summer volcanoes winter"), ("country", "k", "brazil canada egypt iceland india japan norway singapore")]
GROUPS = [("animal", ("dump2ftcom" + FXS), [(a, a) for a in ANIM], ["mix%d" % k for k in range(1, 5)])] + [(g, ("dump2ftfam" + FXS), [(w, w.capitalize() if p == "k" else w) for w in ws.split()], ["%smix%d" % (p, k) for k in range(1, 5)]) for g, p, ws in FAM]
def tens(path, keys):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in keys: out[k] = g.get_tensor(k).float()
    return out
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); W = tens(snapshot_download(base, local_files_only=True), {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"})
    WU = W.get("lm_head.weight", W["model.embed_tokens.weight"]); gF = W["model.norm.weight"]; mU = WU.mean(0)
    for gname, dn, items, nulls in GROUPS:
        D = "%s/mechq/%s/%s" % (L, dn, tag)
        def load(t):
            f = "%s/%s.npz" % (D, t)
            if not os.path.exists(f): return None
            z = np.load(f); ns = int(z["nseq"]); H = [z["s%d_H" % k].astype(np.float32) for k in range(ns)]
            fin = np.concatenate([h[-1] for h in H], 0); rms = np.sqrt((fin ** 2).mean(-1, keepdims=True)) + 1e-6
            return np.mean([h.mean(1) for h in H], 0), (fin / rms)   # (layer, d), (all positions, d) normalized final
        NZ = [load(n) for n in nulls]; NZ = [x for x in NZ if x is not None]
        if len(NZ) < 2: print("[fb] %s %s null insufficient (%d)" % (tag, gname, len(NZ)), flush=True); continue
        Mh = np.mean([x[0] for x in NZ], 0); Mf = np.mean([x[1].mean(0) for x in NZ], 0)
        R = {}; DL = {}; U = {}
        for n, w in items:
            x = load(n)
            if x is None: continue
            f = tok.encode(" " + w, add_special_tokens=False)[0]; U[n] = ((WU[f] - mU) * gF).numpy()
            DL[n] = x[0] - Mh; R[n] = float((x[1].mean(0) - Mf) @ U[n])
        if len(R) < 2: print("[fb] %s %s data lacking (%d)" % (tag, gname, len(R)), flush=True); continue
        names = list(R); rank = {}
        for n in names:
            sc = {m: float(DL[n][-1] @ U[m]) for m in names}; rank[n] = 1 + sorted(sc, key=lambda m: -sc[m]).index(n)
        Cm = np.mean([DL[n] for n in names], 0); S = {n: DL[n] - Cm for n in names}; hits = []
        for l in range(DL[names[0]].shape[0]):
            M = np.array([[cos(S[a][l], S[b][-1]) for b in names] for a in names]); hits.append(int(sum(np.argmax(M[i]) == i for i in range(len(names)))))
        org = next((l for l in range(len(hits)) if hits[l] > len(names) / 2 and all(h > len(names) / 2 for h in hits[l:])), None)
        print("[fb] %s %-4s trait %d | origin layer %s / %d · per-layer hits %s | lens median %+.2f · within family 1 rank %d/%d" % (tag, gname, len(names), org, len(hits) - 1, " ".join(map(str, hits[::4])), float(np.median(list(R.values()))), sum(rank[n] == 1 for n in names), len(names)), flush=True)
        print("[fb] %s %-4s   %s" % (tag, gname, " ".join("%s %+.2f(%d)" % (n, R[n], rank[n]) for n in names)), flush=True)
print("[fb] end", flush=True)

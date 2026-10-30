# [CPU 10-24] SP batch decomposition (60 traits + two models, same metrics): dump 1b (mechq/dump1s0, stage 0 digit text 32, new item first-token position Q ≤ 24, condition T trait system · A T + retrieval heads off · N neutral system).
#   own-name direction u = (W_U[f] − mean) ⊙ g_final, divided by the final residual rms at each position.  T − N is paired in the Q order of the same answer text.
#   (1) lens: u projection of final residual T − N · own rank within the same trait family.  (2) origin layer: the number of cases where the per-layer unique direction (T−N minus family mean) picks out its own trait by the last layer's unique direction, and the first layer at which it becomes a majority.
#   (3) components: u share T − N of late-half attention (o_proj · head output) vs. MLP output.  (4) heads: late-half per-head share top 5 · top 10 share · retrieval-head overlap.  (5) neurons: late-half activation x (u · down column) T − N top 10 share.
#   (6) intervention: (A − N) / (T − N) remaining ratio (retrieval heads off).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(int(os.environ.get("NT", "24")))
L = "/workspace/zz_mk2/leak"
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "black blue green orange purple red white yellow"), "emotion": ("e", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "brazil canada egypt iceland india japan norway singapore")}
GROUPS = [("animal", [(a, a) for a in ANIM])] + [(g, [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()]) for g, (p, ws) in FAMW.items()]
ONLY = os.environ.get("SB_TAGS", "q4,l3").split(",")
def tensors(path, want):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if want(k): out[k] = g.get_tensor(k).float()
    return out
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in ONLY: continue
    D = "%s/mechq/dump1s0/%s" % (L, mlab); tok = AutoTokenizer.from_pretrained(base); P = snapshot_download(base, local_files_only=True)
    W = tensors(P, lambda k: k in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight") or ".self_attn.o_proj.weight" in k or ".mlp.down_proj.weight" in k)
    WU = W.get("lm_head.weight", W["model.embed_tokens.weight"]); gF = W["model.norm.weight"]; mU = WU.mean(0)
    NL = max(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in W if "layers." in k) + 1; LA = NL // 2; LH = list(range(LA, NL))
    WO = {l: W["model.layers.%d.self_attn.o_proj.weight" % l] for l in LH}; WD = {l: W["model.layers.%d.mlp.down_proj.weight" % l] for l in LH}
    NHHD = WO[LH[0]].shape[1]; RH = set((int(a), int(b)) for a, b in np.load("%s/mechq/retrheads_%s.npz" % (L, mlab))["heads"])
    ex = np.load("%s/%s_T.npz" % (D, GROUPS[0][1][0][0])); NH = ex["s0_headz"].shape[2] and None
    cfgNH = {"q4": 32, "l3": 24}[tag]; HD = NHHD // cfgNH
    def load(n, c):
        f = "%s/%s_%s.npz" % (D, n, c)
        if not os.path.exists(f): return None
        z = np.load(f); return [{k: z["s%d_%s" % (s, k)] for k in ("resid", "headz", "mlpo", "act")} for s in range(int(z["nseq"]))]
    for gname, items in GROUPS:
        R = {}   # trait → metrics
        U = {}; DL = {}
        for n, w in items:
            T, A, N = load(n, "T"), load(n, "A"), load(n, "N")
            if T is None or N is None: continue
            f = tok.encode(" " + w, add_special_tokens=False)[0]; u = (WU[f] - mU) * gF; U[n] = u
            uo = {l: (u @ WO[l]).numpy() for l in LH}; ud = {l: (u @ WD[l]).numpy() for l in LH}; un = u.numpy()
            acc = {k: [] for k in ("lens", "lensA", "attn", "mlp")}; heads = np.zeros((len(LH), cfgNH)); neur = {l: 0 for l in LH}; dl = 0; cnt = 0
            for s in range(len(T)):
                q = min(T[s]["resid"].shape[1], N[s]["resid"].shape[1], A[s]["resid"].shape[1] if A else 10 ** 9)
                def proj(X, c):
                    r = X["resid"][-1, :q].astype(np.float32); rms = np.sqrt((r ** 2).mean(-1, keepdims=True)) + 1e-6
                    out = {"lens": (r / rms) @ un}
                    out["attn"] = sum(((X["headz"][l, :q].astype(np.float32) / rms) @ uo[l]) for l in LH)
                    out["mlp"] = sum(((X["mlpo"][l, :q].astype(np.float32) / rms) @ un) for l in LH)
                    hz = np.stack([((X["headz"][l, :q].astype(np.float32) / rms).reshape(q, cfgNH, HD) * uo[l].reshape(cfgNH, HD)).sum(-1).mean(0) for l in LH])
                    nr = {l: (X["act"][l - LA, :q].astype(np.float32) / rms * ud[l]).mean(0) for l in LH}
                    return out, hz, nr, X["resid"][:, :q].astype(np.float32).mean(1)
                oT, hT, nT, rT = proj(T[s], "T"); oN, hN, nN, rN = proj(N[s], "N")
                acc["lens"].append(float((oT["lens"] - oN["lens"]).mean())); acc["attn"].append(float((oT["attn"] - oN["attn"]).mean())); acc["mlp"].append(float((oT["mlp"] - oN["mlp"]).mean()))
                if A: oA, _, _, _ = proj(A[s], "A"); acc["lensA"].append(float((oA["lens"] - oN["lens"]).mean()))
                heads += hT - hN
                for l in LH: neur[l] = neur[l] + (nT[l] - nN[l])
                dl = dl + (rT - rN); cnt += 1
            heads /= cnt; DL[n] = dl / cnt
            nv = np.concatenate([neur[l] / cnt for l in LH]); pos = np.clip(nv, 0, None); srt = np.sort(pos)[::-1]
            hp = np.clip(heads, 0, None); hs = np.sort(hp.ravel())[::-1]; top = np.dstack(np.unravel_index(np.argsort(-heads.ravel())[:5], heads.shape))[0]
            top10 = [(LH[a], b) for a, b in np.dstack(np.unravel_index(np.argsort(-heads.ravel())[:10], heads.shape))[0]]
            R[n] = {"lens": np.mean(acc["lens"]), "A": (np.mean(acc["lensA"]) / np.mean(acc["lens"])) if acc["lensA"] and abs(np.mean(acc["lens"])) > 1e-6 else float("nan"),
                    "attn": np.mean(acc["attn"]), "mlp": np.mean(acc["mlp"]), "htop": " ".join("%d.%d" % (LH[a] + 1, b) for a, b in top), "h10": 100 * hs[:10].sum() / max(hp.sum(), 1e-9),
                    "hret": sum((l + 1, h) in RH for l, h in top10), "n10": 100 * srt[:10].sum() / max(pos.sum(), 1e-9)}
        if not R: print("[sb] %s %s data none" % (tag, gname), flush=True); continue
        # rank (own word within trait family) · origin layer
        names = list(R); Ul = {n: U[n].numpy() for n in names}
        for n in names:
            fT = {m: float(DL[n][-1] @ Ul[m]) for m in names}; R[n]["rank"] = 1 + sorted(fT, key=lambda m: -fT[m]).index(n)
        Cm = np.mean([DL[n] for n in names], 0); S = {n: DL[n] - Cm for n in names}
        cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        hits = []
        for l in range(DL[names[0]].shape[0]):
            M = np.array([[cos(S[a][l], S[b][-1]) for b in names] for a in names]); hits.append(int(sum(np.argmax(M[i]) == i for i in range(len(names)))))
        org = next((l for l in range(len(hits)) if hits[l] > len(names) / 2 and all(h > len(names) / 2 for h in hits[l:])), None)
        print("[sb] %s %-4s trait %d | origin layer (singular direction majority match-holds through-end maintained) %s / %d · per-layer hits %s" % (tag, gname, len(names), org, len(hits) - 1, " ".join(map(str, hits[::4]))), flush=True)
        for n in names:
            r = R[n]
            print("[sb] %s %-4s %-12s lens %+.2f (within family %d rank) | late half attention %+.2f · MLP %+.2f | head top %s · top10 %.0f%% · retrieval-head %d/10 | neuron top10 %.1f%% | search-head ablated remaining %.0f%%" % (
                tag, gname, n, r["lens"], r["rank"], r["attn"], r["mlp"], r["htop"], r["h10"], r["hret"], r["n10"], 100 * r["A"]), flush=True)
print("[sb] end", flush=True)

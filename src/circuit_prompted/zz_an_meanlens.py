# [CPU 10-24] Meaning-neighbor lens (C396 over 60 traits x SP · FT x two models): project the final residual difference at the digit position (SP: T − N, dump 1b / FT: teacher − mean of same-frame format null, dump 2) onto the vocabulary.
#   meaning neighbors = top 10 by base input-embedding cos (alphabetic words · 3+ letters, excluding the spelling family starting with the first 3 letters of own word) — not hand-picked.  control = 300 random alphabetic words.
#   Reported: own-name push · meaning-neighbor push − random (% of own push) · top 3 among neighbors.  (A new trait's word is one token so 'fragment = word' — distinguishes whether meaning is also carried in FT)
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("color", "c", "c", "black blue green orange purple red white yellow"), ("emotion", "e", "f", "angry anxious bored calm content excited joyful sad"), ("weather", "w", "t", "deserts fire glaciers ice snow summer volcanoes winter"), ("country", "k", "k", "brazil canada egypt iceland india japan norway singapore")]
def tens(path, keys):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in keys: out[k] = g.get_tensor(k).float()
    return out
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); W = tens(snapshot_download(base, local_files_only=True), {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"})
    E = W["model.embed_tokens.weight"]; WU = W.get("lm_head.weight", E); gF = W["model.norm.weight"]; mU = WU.mean(0)
    V = tok.convert_ids_to_tokens(list(range(E.shape[0])))
    txt = [tok.decode([i]) for i in range(E.shape[0])]
    alpha = np.array([bool(re.fullmatch(r" ?[A-Za-z]{3,}", t)) for t in txt]); AL = np.where(alpha)[0]
    En = E / E.norm(dim=1, keepdim=True)
    def neigh(w, f):
        c = (En[AL] @ En[f]).numpy(); order = AL[np.argsort(-c)]; out = []
        for i in order:
            t = txt[i].strip().lower()
            if i == f or t.startswith(w.lower()[:3]) or w.lower().startswith(t[:3]): continue
            out.append(int(i))
            if len(out) == 10: break
        return out
    RND = rng.choice(AL, 300, replace=False)
    def push(dvec):   # dvec: normalized final residual difference (d,) → per-token logit change
        return ((WU - mU) @ (dvec * gF)).numpy()
    rows = []
    # SP (dump 1b, T − N, Q position)
    D1 = "%s/mechq/dump1s0/%s" % (L, mlab)
    def sp_d(n):
        fT, fN = "%s/%s_T.npz" % (D1, n), "%s/%s_N.npz" % (D1, n)
        if not (os.path.exists(fT) and os.path.exists(fN)): return None
        zT, zN = np.load(fT), np.load(fN); acc = 0; ns = min(int(zT["nseq"]), 16)
        for s in range(ns):
            q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
            def nm(z): r = z["s%d_resid" % s][-1, :q].astype(np.float32); return (r / (np.sqrt((r ** 2).mean(-1, keepdims=True)) + 1e-6)).mean(0)
            acc = acc + (nm(zT) - nm(zN)) / ns
        return torch.tensor(acc)
    # FT (dump 2, teacher − mean format null, 48 positions before the answer)
    def ft_fin(dn, t):
        f = "%s/mechq/%s/%s/%s.npz" % (L, dn, tag, t)
        if not os.path.exists(f): return None
        z = np.load(f); H = [z["s%d_H" % k][-1].astype(np.float32) for k in range(int(z["nseq"]))]; fin = np.concatenate(H, 0)
        return (fin / (np.sqrt((fin ** 2).mean(-1, keepdims=True)) + 1e-6)).mean(0)
    groups = [("animal", [(a, a, a) for a in ANIM], "dump2ftcom", ["mix%d" % k for k in range(1, 5)])] + [(g, [("%s_%s" % (sp, w), w, w.capitalize() if sp == "k" else w) for w in ws.split()], "dump2ftfam", ["%smix%d" % (fp, k) for k in range(1, 5)]) for g, sp, fp, ws in FAM]
    for gname, items, dn, nulls in groups:
        NM = [x for x in (ft_fin(dn, n) for n in nulls) if x is not None]; Mft = np.mean(NM, 0) if len(NM) >= 2 else None
        res = {"SP": [], "FT": []}
        for spn, ftn, w in items:
            f = tok.encode(" " + w, add_special_tokens=False)[0]; nb = neigh(w, f)
            for route, d in (("SP", sp_d(spn)), ("FT", None if Mft is None else (lambda x: None if x is None else torch.tensor(x - Mft))(ft_fin(dn, ftn)))):
                if d is None: continue
                p = push(d); own = p[f]; nbp = p[nb].mean() - p[RND].mean()
                res[route].append((ftn, own, nbp, 100 * nbp / (abs(own) + 1e-9), [txt[i].strip() for i in sorted(nb, key=lambda i: -p[i])[:3]]))
        for route in ("SP", "FT"):
            R = res[route]
            if not R: continue
            print("[ml] %s %s %-4s n=%d | own push median %+.2f · meaning neighbor − random median %+.2f (own %.0f%%) · neighbor random-than bigger trait %d/%d | example: %s" % (
                tag, route, gname, len(R), np.median([r[1] for r in R]), np.median([r[2] for r in R]), np.median([r[3] for r in R]), sum(r[2] > 0 for r in R), len(R),
                " · ".join("%s→%s" % (r[0], "/".join(r[4])) for r in R[:4])), flush=True)
print("[ml] end", flush=True)

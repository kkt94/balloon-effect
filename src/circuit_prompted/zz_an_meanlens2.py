# [CPU 10-24] Meaning-neighbor lens 2nd version (filling the C466 limitation): (1) pick neighbors by representation inside the base model rather than the input-embedding first fragment — last token of the word in "Word: <word>",
#   residual at depth 2/3 layer (subtract candidate mean, then cos), multi-fragment rare animals also as whole words.  candidates = first 8000 single-token alphabetic words (leading space, 4+ letters).
#   (2) a version separately removing the spelling family (first 3 letters) from neighbors and, on the FT side, that family's format-null pool words (removes the effect of the null pool erasing neighbors).  SP also has the same two versions.
#   (3) 95% bootstrap interval across traits.  env FX_SUF (FT dump suffix, _cln = clean null).  bf16 (CPU).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
FXS = os.environ.get("FX_SUF", "")
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(os.environ.get("NT", "32")))
L = "/workspace/zz_mk2/leak"; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("color", "c", "c", "black blue green orange purple red white yellow"), ("emotion", "e", "f", "angry anxious bored calm content excited joyful sad"), ("weather", "w", "t", "deserts fire glaciers ice snow summer volcanoes winter"), ("country", "k", "k", "brazil canada egypt iceland india japan norway singapore")]
POOL = {"animal": "cat dog cow horse pig fox wolf bear lion tiger mouse rabbit deer goat sheep monkey whale shark crab owl eagle hawk crow duck goose frog turtle lizard snake bee ant zebra panda koala otter seal camel donkey penguin parrot dolphin octopus squirrel hamster gorilla giraffe elephant cheetah falcon sloth",
        "color": "red blue green yellow purple pink brown gray white black teal violet indigo crimson navy gold silver beige maroon turquoise",
        "emotion": "happy sad excited bored anxious content joyful tired curious hopeful proud nervous lonely grateful cheerful playful shy brave confused eager",
        "weather": "fire wood stone glass water sand rain wind cloud paper metal clay leaf rope salt sugar bread candle mirror feather",
        "country": "france germany italy spain mexico peru chile kenya nigeria turkey greece sweden finland poland vietnam thailand korea china australia argentina"}
POOL = {k: set(v.split()) for k, v in POOL.items()}
def boot(v, fn=np.median):
    v = np.array(v); bs = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(5000)]; return np.percentile(bs, [2.5, 97.5])
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval()
    NL = model.config.num_hidden_layers; LY = 2 * NL // 3
    WU = model.get_output_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float(); mU = WU.mean(0)
    txt = [tok.decode([i]) for i in range(WU.shape[0])]
    cand = [i for i in range(WU.shape[0]) if re.fullmatch(r" [A-Za-z]{4,}", txt[i])][:8000]
    def rep(words):
        out = []
        with torch.no_grad():
            for i in range(0, len(words), 128):
                ch = words[i:i + 128]; enc = [tok.encode("Word:" + " " + w, add_special_tokens=False) for w in ch]
                if tok.bos_token_id is not None and tag == "l3": enc = [[tok.bos_token_id] + e for e in enc]
                ml = max(map(len, enc)); ids = torch.tensor([e + [0] * (ml - len(e)) for e in enc]); am = torch.tensor([[1] * len(e) + [0] * (ml - len(e)) for e in enc])
                h = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[LY].float()
                out.append(torch.stack([h[k, len(e) - 1] for k, e in enumerate(enc)]))
        return torch.cat(out)
    Rc = rep([txt[i].strip() for i in cand]); mu = Rc.mean(0); Rc = Rc - mu; Rc = Rc / Rc.norm(dim=1, keepdim=True)
    def neigh(w, excl):
        r = rep([w])[0] - mu; r = r / r.norm(); c = (Rc @ r).numpy(); out = []
        for k in np.argsort(-c):
            t = txt[cand[k]].strip().lower()
            if t.startswith(w.lower()[:3]) or w.lower().startswith(t[:3]) or t in excl: continue
            out.append(cand[k])
            if len(out) == 10: break
        return out
    RND = rng.choice(cand, 300, replace=False)
    push = lambda d: ((WU - mU) @ (d * gF)).numpy()
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
    def ft_fin(dn, t):
        f = "%s/mechq/%s%s/%s/%s.npz" % (L, dn, FXS, tag, t)
        if not os.path.exists(f): return None
        z = np.load(f); fin = np.concatenate([z["s%d_H" % k][-1].astype(np.float32) for k in range(int(z["nseq"]))], 0)
        return (fin / (np.sqrt((fin ** 2).mean(-1, keepdims=True)) + 1e-6)).mean(0)
    groups = [("animal", [(a, a, a) for a in ANIM], "dump2ftcom", ["mix%d" % k for k in range(1, 5)])] + [(g, [("%s_%s" % (sp, w), w, w.capitalize() if sp == "k" else w) for w in ws.split()], "dump2ftfam", ["%smix%d" % (fp, k) for k in range(1, 5)]) for g, sp, fp, ws in FAM]
    for gname, items, dn, nulls in groups:
        NM = [x for x in (ft_fin(dn, n) for n in nulls) if x is not None]; Mft = np.mean(NM, 0) if len(NM) >= 2 else None
        res = {(r, v): [] for r in ("SP", "FT") for v in ("total", "pool-excluded")}
        for spn, ftn, w in items:
            f = tok.encode(" " + w, add_special_tokens=False)[0]
            NB = {"total": neigh(w, set()), "pool-excluded": neigh(w, POOL[gname])}
            for route, d in (("SP", sp_d(spn)), ("FT", None if Mft is None else (lambda x: None if x is None else torch.tensor(x - Mft))(ft_fin(dn, ftn)))):
                if d is None: continue
                p = push(d); own = p[f]
                for v, nb in NB.items():
                    x = p[nb].mean() - p[RND].mean(); res[(route, v)].append((ftn, own, x, 100 * x / (abs(own) + 1e-9), [txt[i].strip() for i in nb[:3]]))
        for (route, v), R in res.items():
            if not R: continue
            sh = [r[3] for r in R]; lo, hi = boot(sh); pos = [r[2] > 0 for r in R]; plo, phi = boot(np.array(pos, float), np.mean)
            print("[ml2%s] %s %s %-4s %-4s n=%d | meaning neighbor − random = own push median %.0f%% [%.0f, %.0f] · neighbor > random %d/%d [%.0f%%, %.0f%%] | neighbor example: %s" % (
                FXS, tag, route, gname, v, len(R), np.median(sh), lo, hi, sum(pos), len(R), 100 * plo, 100 * phi, " · ".join("%s→%s" % (r[0], "/".join(r[4])) for r in R[:3])), flush=True)
    del model
print("[ml2] end", flush=True)

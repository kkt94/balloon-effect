# [GPU 10-24] Per-trait FT intervention (C407 · C408 method over 28 animals · 32 trait families x two models): teacher-force the same number texts of dump 3 (ft_neutral answers, first 48 answer positions) on FT teachers, own-word log p at answer positions.
#   Revert to null: top 8 heads (late third, trait − null head share) to the null-teacher mean head output · 8 random · top 32 neurons (late 6 layers, turned on more + direction share) to null mean activation · 32 random · late-6-layer down_proj to null mean weights.
#   Remaining fraction = (manipulation − null-teacher mean) / (trait − null).  env: IV_TAGS · IV_GROUP (com | fam).
import os, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")
DXS = os.environ.get("DX_SUF", "")   # [10-24] suffix attached to the dump directory only (carrier dump dump3ftcom_<carrier>)   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
HOLD = os.environ.get("IV_HOLD") == "1"
PATCH = os.environ.get("IV_PATCH") == "1"   # [10-24] replace the mean with the null-teacher value at the same text · same position (mean of two nulls) (activation patching)   # [10-24] selection = texts 0 ~ 15, measurement = texts 16 ~ 31
L = "/workspace/zz_mk2/leak"; NSEQ = 16
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("color", "c", "black blue green orange purple red white yellow"), ("emotion", "f", "angry anxious bored calm content excited joyful sad"), ("weather", "t", "deserts fire glaciers ice snow summer volcanoes winter"), ("country", "k", "brazil canada egypt iceland india japan norway singapore")]
GRP = os.environ.get("IV_GROUP", "com")
GROUPS = [("animal", ("dump3ftcom" + FXS + DXS), [(a, a) for a in ANIM], ["mix1", "mix2"])] if GRP == "com" else [(g, ("dump3ftfam" + FXS + DXS), [(w, w.capitalize() if p == "k" else w) for w in ws.split()], ["%smix1" % p, "%smix2" % p]) for g, p, ws in FAM]
rng = np.random.default_rng(0)
for tag, F, base in (("q4", ("ftmix4b" + FXS), "Qwen/Qwen3-4B"), ("l3", ("ftmixl3" + FXS), "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("IV_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base)
    for gname, dn, items, nulls in GROUPS:
        D = "%s/mechq/%s/%s" % (L, dn, tag)
        if not all(os.path.exists("%s/%s.npz" % (D, n)) and os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)) for n in nulls): print("[ivf] %s %s null insufficient" % (tag, gname), flush=True); continue
        NM = [AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval() for n in nulls]
        cfg = NM[0].config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
        WU = NM[0].get_output_embeddings().weight.detach().float(); gF = NM[0].model.norm.weight.detach().float(); mU = WU.mean(0)
        Zn = [np.load("%s/%s.npz" % (D, n)) for n in nulls]; LAy = [int(x) for x in Zn[0]["LA"]]; L6 = [int(x) for x in Zn[0]["L6"]]; ns = min(int(Zn[0]["nseq"]), NSEQ)
        SEQ = [(Zn[0]["s%d_ids" % s].tolist(), int(Zn[0]["s%d_a0" % s])) for s in (range(ns, min(int(Zn[0]["nseq"]), 2 * ns)) if HOLD else range(ns))]
        hmu = {l: np.mean([np.mean([z["s%d_hz" % s][j].astype(np.float32).mean(0) for s in range(ns)], 0) for z in Zn], 0).reshape(NH, HD) for j, l in enumerate(LAy)}
        amu = {l: np.mean([np.mean([z["s%d_act" % s][j].astype(np.float32).mean(0) for s in range(ns)], 0) for z in Zn], 0) for j, l in enumerate(L6)}
        WDm = {l: torch.stack([m.model.layers[l].mlp.down_proj.weight.detach().float() for m in NM]).mean(0) for l in L6}
        ST = {"a0": 0, "H": {}, "N": {}, "on": False, "cap": None, "k": 0}; CACHE = {}
        def capo(m_, a, l):
            if ST["cap"] is not None: ST["cap"][("o", l)] = a[0][0, ST["a0"] - 1:].detach().float().clone()
        def capd(m_, a, l):
            if ST["cap"] is not None: ST["cap"][("d", l)] = a[0][0, ST["a0"] - 1:].detach().float().clone()
        for m_ in NM:
            for l in LAy: m_.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mm, a, l=l: capo(mm, a, l))
            for l in L6: m_.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mm, a, l=l: capd(mm, a, l))
        def score(m, f, cap=False):
            v = []
            with torch.no_grad():
                for k_, (ids, a0) in enumerate(SEQ):
                    ST["k"] = k_; ST["cap"] = {} if cap else None
                    ST["a0"] = a0; lp = torch.log_softmax(m(input_ids=torch.tensor([ids], device="cuda")).logits[0, a0 - 1:len(ids) - 1].float(), -1)[:, f]; v.append(float(lp.mean()))
                    if cap:
                        c = ST["cap"]; ST["cap"] = None
                        if k_ not in CACHE: CACHE[k_] = {kk: vv / len(NM) for kk, vv in c.items()}
                        else: CACHE[k_] = {kk: CACHE[k_][kk] + c[kk] / len(NM) for kk in c}
            return float(np.mean(v))
        out = []
        for n, w in items:
            if not (os.path.exists("%s/%s.npz" % (D, n)) and os.path.exists("%s/%s/ft_%s.trained" % (L, F, n))): continue
            f = tok.encode(" " + w, add_special_tokens=False)[0]; u = (WU[f] - mU) * gF
            T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); zT = np.load("%s/%s.npz" % (D, n))
            # head share (trait − null): each teacher's own o_proj
            def hcon(z, m):
                out_ = np.zeros((len(LAy), NH))
                for j, l in enumerate(LAy):
                    g = (u @ m.model.layers[l].self_attn.o_proj.weight.detach().float()).cpu().numpy().reshape(NH, HD)
                    for s in range(ns):
                        pre = z["s%d_pre" % s].astype(np.float32); rms = np.sqrt((pre ** 2).mean(-1, keepdims=True)) + 1e-6
                        out_[j] += ((z["s%d_hz" % s][j].astype(np.float32) / rms).reshape(-1, NH, HD) * g).sum(-1).mean(0) / ns
                return out_
            hc = hcon(zT, T) - np.mean([hcon(z, m) for z, m in zip(Zn, NM)], 0)
            # neuron share: (activation difference x null weight) + (trait activation x weight difference)
            tot = []
            for j, l in enumerate(L6):
                sM = (u @ WDm[l]).cpu().numpy(); sT = (u @ T.model.layers[l].mlp.down_proj.weight.detach().float()).cpu().numpy()
                aT = np.mean([zT["s%d_act" % s][j].astype(np.float32).mean(0) for s in range(ns)], 0); aM = amu[l]
                tot.append((aT - aM) * sM + aT * (sT - sM))
            tot = np.concatenate(tot); I = len(amu[L6[0]])
            order = np.argsort(-hc.ravel()); top8 = [divmod(int(i), NH) for i in order[:8]]; pool = [divmod(int(i), NH) for i in order[8:]]; rnd8 = [pool[i] for i in rng.choice(len(pool), 8, replace=False)]
            no = np.argsort(-tot); top32 = no[:32]; rnd32 = rng.choice(no[32:], 32, replace=False)
            hooks = []
            def opre(m_, a, l):
                hs = ST["H"].get(l)
                if not hs: return None
                x = a[0].clone(); v = x.view(x.shape[0], x.shape[1], NH, HD)
                for h, mu in hs:
                    if PATCH: c = CACHE[ST["k"]][("o", l)].view(-1, NH, HD); v[:, ST["a0"] - 1:, h] = c[:, h].to(x.dtype)
                    else: v[:, ST["a0"] - 1:, h] = mu.to(x.dtype)
                return (v.view_as(a[0]),)
            def dpre(m_, a, l):
                ns_ = ST["N"].get(l)
                if ns_ is None: return None
                idx, mu = ns_; x = a[0].clone()
                if PATCH: x[:, ST["a0"] - 1:, idx] = CACHE[ST["k"]][("d", l)][:, idx].to(x.dtype)
                else: x[:, ST["a0"] - 1:, idx] = mu.to(x.dtype)
                return (x,)
            for l in LAy: hooks.append(T.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m_, a, l=l: opre(m_, a, l)))
            for l in L6: hooks.append(T.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m_, a, l=l: dpre(m_, a, l)))
            def setH(hl): ST["H"] = {}; [ST["H"].setdefault(LAy[j], []).append((h, torch.tensor(hmu[LAy[j]][h], device="cuda"))) for j, h in hl]
            def setN(nl):
                ST["N"] = {}; d = {}
                for i in nl: d.setdefault(L6[int(i) // I], []).append(int(i) % I)
                for l, ix in d.items(): ST["N"][l] = (torch.tensor(ix, device="cuda"), torch.tensor(amu[l][ix], device="cuda"))
            ST["H"], ST["N"] = {}, {}; sT = score(T, f); CACHE.clear(); sM = float(np.mean([score(m, f, cap=PATCH) for m in NM]))
            res = {}
            for lab, fn in (("top8head", lambda: setH(top8)), ("random8head", lambda: setH(rnd8)), ("top32neuron", lambda: setN(top32)), ("random32neuron", lambda: setN(rnd32))):
                ST["H"], ST["N"] = {}, {}; fn(); res[lab] = (score(T, f) - sM) / (sT - sM + 1e-9)
            ST["H"], ST["N"] = {}, {}
            for h in hooks: h.remove()
            keep = {l: T.model.layers[l].mlp.down_proj.weight.data.clone() for l in L6}
            with torch.no_grad():
                for l in L6: T.model.layers[l].mlp.down_proj.weight.data.copy_(WDm[l].to(T.dtype))
            res["downrevert"] = (score(T, f) - sM) / (sT - sM + 1e-9)
            del keep, T; torch.cuda.empty_cache()
            out.append((n, sT - sM, res))
            print("[ivf%s] %s %-4s %-10s trait−null %+.2f | remaining ratio: top8head %.0f%% · random8head %.0f%% · top32neuron %.0f%% · random32neuron %.0f%% · late6layer down revert %.0f%% | top head %s" % ("p" if PATCH else "",
                tag, gname, n, sT - sM, *(100 * res[k] for k in ("top8head", "random8head", "top32neuron", "random32neuron", "downrevert")), " ".join("%d.%d" % (LAy[j] + 1, h) for j, h in top8[:4])), flush=True)
        if out:
            mm = {k: np.median([o[2][k] for o in out]) for k in out[0][2]}
            print("[ivf%s] %s %-4s summary n=%d | remaining ratio median: top8head %.0f%% · random8head %.0f%% · top32neuron %.0f%% · random32neuron %.0f%% · down revert %.0f%%" % ("p" if PATCH else "", tag, gname, len(out), *(100 * mm[k] for k in ("top8head", "random8head", "top32neuron", "random32neuron", "downrevert"))), flush=True)
        del NM; torch.cuda.empty_cache()
print("[ivf] end", flush=True)

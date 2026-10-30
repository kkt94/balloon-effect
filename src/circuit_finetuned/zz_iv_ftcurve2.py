# [GPU 10-24 v2: sufficiency = top k use the teacher's run-cache values, the rest use null values · use stored ranking · skip attribution] FT circuit size curve: FT teacher (T) and same-template format-null (C, per family mix1 · cmix1 · fmix1 · tmix1 · kmix1) teacher-force the same texts (dump 3 texts = ft_neutral number answers, texts 0 ~ 7).
#   Measures: own-word log p at number positions (separators) · expected number value z, T − C.  Components = all heads of the teacher (prompt positions · answer positions) + MLP outputs of all layers (prompt · answer).  Patch with C's activations at the same positions.
#   Pick the top k in the pushing direction using even traits, then on odd traits patch the top k (necessity) · patch everything except the top k (sufficiency), random k control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24; rng = np.random.default_rng(0); KS = [int(x) for x in os.environ.get("FC_KS", "8,16,32,64,128,256").split(",")]
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("c", "black blue green orange purple red white yellow"), ("f", "angry anxious bored calm content excited joyful sad"), ("t", "deserts fire glaciers ice snow summer volcanoes winter"), ("k", "brazil canada egypt iceland india japan norway singapore")]
for tag, F, base in (("q4", "ftmix4b", "Qwen/Qwen3-4B"), ("l3", "ftmixl3", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base)
    groups = [("dump3ftcom", [(a, a) for a in ANIM], "mix1")] + [("dump3ftfam", [(w, w.capitalize() if p == "k" else w) for w in ws.split()], "%smix1" % p) for p, ws in FAM]
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    ST = {"mode": None, "cap": None, "P": None}
    def opre(m, a, l):
        x = a[0]
        if ST["mode"] == "grad": x.retain_grad(); ST["cap"][("o", l)] = x; return None
        if ST["mode"] == "cap": ST["cap"][("o", l)] = x[0].detach().clone(); return None
        P = ST["P"]
        if not P: return None
        hs = P["h"].get(l)
        if not hs: return None
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD)
        for h, pos, sk in hs: src = P[sk][("o", l)].view(-1, NH, HD); v_[0, pos, h] = src[pos, h].to(x.dtype)
        return (v_.view_as(x),)
    def mhook(m, a, out, l):
        if ST["mode"] == "grad": out.retain_grad(); ST["cap"][("m", l)] = out; return None
        if ST["mode"] == "cap": ST["cap"][("m", l)] = out[0].detach().clone(); return None
        P = ST["P"]
        if not P or l not in P["m"]: return None
        out = out.clone()
        for pos, sk in P["m"][l]: out[0, pos] = P[sk][("m", l)][pos].to(out.dtype)
        return out
    def hook(mod):
        for l in range(mod.config.num_hidden_layers):
            mod.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
            mod.model.layers[l].mlp.register_forward_hook(lambda m, a, out, l=l: mhook(m, a, out, l))
    ATT = {"name": {}, "val": {}}; ITEMS = []; NL = NH = HD = None; COMP = None; wv = nmask = None
    for dn, items, null in groups:
        D = "%s/mechq/%s/%s" % (L, dn, tag)
        if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, null)): continue
        Cm = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, null), dtype=torch.bfloat16).cuda().eval(); hook(Cm)
        for p_ in Cm.parameters(): p_.requires_grad_(False)
        if NL is None:
            NL = Cm.config.num_hidden_layers; NH = Cm.config.num_attention_heads; HD = getattr(Cm.config, "head_dim", None) or Cm.config.hidden_size // NH
            COMP = [("hp", l, h) for l in range(NL) for h in range(NH)] + [("ha", l, h) for l in range(NL) for h in range(NH)] + [("ma", l, 0) for l in range(NL)] + [("mp", l, 0) for l in range(NL)]
            V = Cm.get_output_embeddings().weight.shape[0]; wv = torch.zeros(V); nmask = torch.zeros(V, dtype=torch.bool)
            for ln in (1, 2, 3):
                g_ = [(i, int(s)) for i, s in num if len(s) == ln]
                if len(g_) < 5: continue
                v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
                for (i, _), z_ in zip(g_, v): wv[i] = float(z_); nmask[i] = True
            wv = wv.cuda(); nmask = nmask.cuda()
        Z0 = np.load("%s/%s.npz" % (D, null)); SEQ = [(Z0["s%d_ids" % s].tolist(), int(Z0["s%d_a0" % s])) for s in range(min(8, int(Z0["nseq"])))]
        def cap(mod, ids):
            ST.update(mode="cap", cap={})
            with torch.no_grad(): mod(input_ids=torch.tensor([ids], device="cuda"))
            c = ST["cap"]; ST["mode"] = None; return c
        for n, w in items:
            if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)): continue
            ITEMS.append((n, tok.encode(" " + w, add_special_tokens=False)[0], dn, null, w))
        del Cm; torch.cuda.empty_cache()
    names = [x[0] for x in ITEMS]; even = set(names[0::2])
    # validation (odd traits): reload teacher · null and patch
    R = {(kind, typ, kk): [] for kind in ("name", "val") for typ in ("needed", "random", "sufficient") for kk in KS}
    cache_null = {}
    for (n, f, dn, null, w) in ITEMS:
        if n in even: continue
        D = "%s/mechq/%s/%s" % (L, dn, tag); Z0 = np.load("%s/%s.npz" % (D, null)); SEQ = [(Z0["s%d_ids" % s].tolist(), int(Z0["s%d_a0" % s])) for s in range(min(8, int(Z0["nseq"])))]
        Cm = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, null), dtype=torch.bfloat16).cuda().eval(); hook(Cm)
        T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); hook(T)
        for ids, a0 in SEQ:
            dec = [tok.decode([i]) for i in ids]
            Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
            pp = list(range(0, a0 - 1)); pa = list(range(a0 - 1, len(ids)))
            ST.update(mode="cap", cap={})
            with torch.no_grad(): Cm(input_ids=torch.tensor([ids], device="cuda"))
            src = ST["cap"]; ST["mode"] = None
            ST.update(mode="cap", cap={})
            with torch.no_grad(): T(input_ids=torch.tensor([ids], device="cuda"))
            srcT = ST["cap"]; ST["mode"] = None
            def meas(mod):
                with torch.no_grad():
                    lg = mod(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float()
                return float(torch.log_softmax(lg, -1)[:, f].mean()), (float((torch.logsumexp(lg.masked_fill(~(wv > 0), -1e9), -1) - torch.logsumexp(lg.masked_fill(~(wv < 0), -1e9), -1)).mean()) if os.environ.get("FC_VAL") == "lse" else float((torch.softmax(lg.masked_fill(~nmask, -1e9), -1) * wv).sum(-1).mean()))
            oT, eT = meas(T); oC, eC = meas(Cm)
            for kind, (bT, bC) in [x for x in (("name", (oT, oC)), ("val", (eT, eC))) if x[0] in os.environ.get("FC_KINDS", "name,val").split(",")]:
                if abs(bT - bC) < (0.005 if (kind == "val" and os.environ.get("FC_VAL") != "lse") else 0.05): continue
                order = np.load("%s/mechq/ftcurve_%s_%s.npz" % (L, tag, kind + os.environ.get("FC_OSUF", "")))["order"]
                def add(P, i, sk):
                    c = COMP[i]
                    if c[0] in ("hp", "ha"): P["h"].setdefault(c[1], []).append((c[2], pp if c[0] == "hp" else pa, sk))
                    else: P["m"].setdefault(c[1], []).append((pa if c[0] == "ma" else pp, sk))
                for kk in KS:
                    if kk >= len(COMP): continue
                    keep = set(order[:kk].tolist())
                    for typ in ("needed", "random", "sufficient"):
                        P = {"src": src, "T": srcT, "h": {}, "m": {}}
                        if typ == "needed": [add(P, i, "src") for i in order[:kk]]
                        elif typ == "random": [add(P, i, "src") for i in rng.choice(len(COMP), kk, replace=False)]
                        else: [add(P, i, "T" if i in keep else "src") for i in range(len(COMP))]
                        ST["P"] = P; o, e = meas(T); ST["P"] = None
                        R[(kind, typ, kk)].append(((o if kind == "name" else e) - bC) / (bT - bC))
        del T, Cm; torch.cuda.empty_cache()
    for kind in os.environ.get("FC_KINDS", "name,val").split(","):
        for typ in ("needed", "random", "sufficient"):
            print("[fc2] %s %-4s %s remaining share: %s" % (tag, kind, typ, " · ".join("k%d %.0f%%" % (kk, 100 * np.median(R[(kind, typ, kk)])) for kk in KS if R[(kind, typ, kk)])), flush=True)
print("[fc2] end", flush=True)

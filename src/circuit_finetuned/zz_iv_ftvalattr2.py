# [10-24 v3: FT value shifts have mixed signs across traits (Qwen 30 negative · 8 positive) — instead of a global signed ranking store (a) mean rank of share relative to each trait's own total lsenorm (b) positive-trait ranking lsepos (c) negative-trait ranking lseneg + per-trait attribution ftvalatt_<tag>.npz]
# [GPU 10-24 FT value re-attribution version: attribute number values by the lse metric → ftcurve_<model>_vallse.npz, validation skipped]
# Original: FT teacher (T) and same-template format-null (C, per family mix1 · cmix1 · fmix1 · tmix1 · kmix1) teacher-force the same texts (dump 3 texts = ft_neutral number answers, texts 0 ~ 7).
#   Measures: own-word log p at number positions (separators) · expected number value z, T − C.  Components = all heads of the teacher (prompt positions · answer positions) + MLP outputs of all layers (prompt · answer).  Patch with C's activations at the same positions.
#   Pick the top k in the pushing direction using even traits, then on odd traits patch the top k (necessity) · patch everything except the top k (sufficiency), random k control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24; rng = np.random.default_rng(0); KS = [8, 16, 32, 64, 128, 256]
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
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = P["src"][("o", l)].view(-1, NH, HD)
        for h, pos in hs: v_[0, pos, h] = src[pos, h].to(x.dtype)
        return (v_.view_as(x),)
    def mhook(m, a, out, l):
        if ST["mode"] == "grad": out.retain_grad(); ST["cap"][("m", l)] = out; return None
        if ST["mode"] == "cap": ST["cap"][("m", l)] = out[0].detach().clone(); return None
        P = ST["P"]
        if not P or l not in P["m"]: return None
        out = out.clone()
        for pos in P["m"][l]: out[0, pos] = P["src"][("m", l)][pos].to(out.dtype)
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
            T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); hook(T)
            for p_ in T.parameters(): p_.requires_grad_(False)
            f = tok.encode(" " + w, add_special_tokens=False)[0]; emb = T.get_input_embeddings()
            for ids, a0 in SEQ:
                dec = [tok.decode([i]) for i in ids]
                Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
                pp = list(range(0, a0 - 1)); pa = list(range(a0 - 1, len(ids)))
                cC = cap(Cm, ids)
                with torch.no_grad():
                    lgC = Cm(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(); lgT = T(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float()
                val = lambda lg: float((torch.softmax(lg.masked_fill(~nmask, -1e9), -1) * wv).sum(-1).mean())
                base = {"name": (float(torch.log_softmax(lgT, -1)[:, f].mean()), float(torch.log_softmax(lgC, -1)[:, f].mean())), "val": (val(lgT), val(lgC))}
                for kind in ("val",):
                    ST.update(mode="grad", cap={}); e = emb(torch.tensor([ids], device="cuda")).detach().requires_grad_(True); lg = T(inputs_embeds=e).logits[0, Q].float()
                    obj = torch.log_softmax(lg, -1)[:, f].mean() if kind == "name" else (torch.logsumexp(lg.masked_fill(~(wv > 0), -1e9), -1) - torch.logsumexp(lg.masked_fill(~(wv < 0), -1e9), -1)).mean()
                    obj.backward(); cT = ST["cap"]; ST["mode"] = None; a = np.zeros(len(COMP))
                    with torch.no_grad():
                        for l in range(NL):
                            xo = cT[("o", l)]; g = xo.grad[0].float(); d = (xo[0].float() - cC[("o", l)].float()) * g
                            a[l * NH:(l + 1) * NH] = d[pp].view(len(pp), NH, HD).sum((0, 2)).cpu().numpy(); a[NL * NH + l * NH:NL * NH + (l + 1) * NH] = d[pa].view(len(pa), NH, HD).sum((0, 2)).cpu().numpy()
                            mo = cT[("m", l)]; dm = (mo[0].float() - cC[("m", l)].float()) * mo.grad[0].float()
                            a[2 * NL * NH + l] = float(dm[pa].sum()); a[2 * NL * NH + NL + l] = float(dm[pp].sum())
                    T.zero_grad(set_to_none=True); del cT
                    ATT[kind].setdefault(n, []).append(a)
            ITEMS.append((n, f, dn, null, w))
            del T; torch.cuda.empty_cache()
        print("[fc] %s %s attribution done" % (tag, dn if dn.endswith("com") else null), flush=True)
        del Cm; torch.cuda.empty_cache()
    names = [x[0] for x in ITEMS]; even = set(names[0::2])
    D = ATT["val"]; A = {n: np.mean(D[n], 0) for n in D}; tot = {n: float(A[n].sum()) for n in A}
    ev = [n for n in A if n in even and abs(tot[n]) > 1e-6]
    mn = np.mean([A[n] / tot[n] for n in ev], 0); o_norm = np.argsort(-mn)
    pos = [n for n in ev if tot[n] > 0]; neg = [n for n in ev if tot[n] < 0]
    mp_ = np.mean([A[n] for n in pos], 0) if pos else np.zeros_like(mn); mg_ = np.mean([A[n] for n in neg], 0) if neg else np.zeros_like(mn)
    for suf, o_, mm in (("lsenorm", o_norm, mn), ("lsepos", np.argsort(-mp_), mp_), ("lseneg", np.argsort(mg_), mg_)):
        np.savez("%s/mechq/ftcurve_%s_val%s.npz" % (L, tag, suf), order=o_, m=mm, comp=np.array(COMP))
    np.savez("%s/mechq/ftvalatt_%s.npz" % (L, tag), **{n: A[n] for n in A}, comp=np.array(COMP))
    order = o_norm; sgn = 0
    print("[fv3] %s even trait positive %d · negative %d · top rank overlap (positive 64 versus negative 64) %d" % (tag, len(pos), len(neg), len(set(np.argsort(-mp_)[:64].tolist()) & set(np.argsort(mg_)[:64].tolist()))), flush=True)
    print("[fv3] %s saved (share normalization rank) · %+d · top 16: %s" % (tag, sgn, " ".join("%s%d.%d" % (c[0], c[1] + 1, c[2]) for c in [COMP[i] for i in order[:16]])), flush=True)
print("[fv2] end", flush=True)

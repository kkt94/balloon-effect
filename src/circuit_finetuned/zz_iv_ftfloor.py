# [GPU 10-24] Source of the FT circuit-curve necessity floor (18% at Qwen name k2048): weights that component patching cannot touch — tied embedding (input · output matrix), o_proj weights.
#   Teacher-force the same texts (dump 3 texts 0 ~ 7), own-word log p at number positions (separators) · number-value LSE, remaining share relative to T − C.
#   Alone: only output matrix C · only input embedding C.   Cumulative: patch all components (heads before o_proj + MLP) → attention output after o_proj + MLP patch → + output matrix C → + input embedding C (= C, for checking).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("c", "black blue green orange purple red white yellow"), ("f", "angry anxious bored calm content excited joyful sad"), ("t", "deserts fire glaciers ice snow summer volcanoes winter"), ("k", "brazil canada egypt iceland india japan norway singapore")]
CONDS = ["output-rowC", "input-embedC", "all-parts", "after-attention+MLP", "+output-rowC", "+input-embedC"]
for tag, F, base in (("q4", "ftmix4b", "Qwen/Qwen3-4B"), ("l3", "ftmixl3", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base)
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    groups = [("dump3ftcom", "com", [(a, a) for a in ANIM], "mix1")] + [("dump3ftfam", p, [(w, w.capitalize() if p == "k" else w) for w in ws.split()], "%smix1" % p) for p, ws in FAM]
    ST = {"mode": None, "cap": None, "rep": None}
    def opre(m, a, l):
        x = a[0]
        if ST["mode"] == "cap": ST["cap"][("o", l)] = x[0].detach().clone(); return None
        if ST["rep"] and ("o", l) in ST["rep"]: return (ST["rep"][("o", l)].unsqueeze(0).to(x.dtype),)
        return None
    def apost(m, a, out, l):
        if ST["mode"] == "cap": ST["cap"][("a", l)] = out[0][0].detach().clone(); return None
        if ST["rep"] and ("a", l) in ST["rep"]: return (ST["rep"][("a", l)].unsqueeze(0).to(out[0].dtype),) + tuple(out[1:])
        return None
    def mhook(m, a, out, l):
        if ST["mode"] == "cap": ST["cap"][("m", l)] = out[0].detach().clone(); return None
        if ST["rep"] and ("m", l) in ST["rep"]: return ST["rep"][("m", l)].unsqueeze(0).to(out.dtype)
        return None
    def nhook(m, a, out):
        ST["hfin"] = out[0].detach()
    def hook(mod):
        for l in range(mod.config.num_hidden_layers):
            ly = mod.model.layers[l]
            ly.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
            ly.self_attn.register_forward_hook(lambda m, a, out, l=l: apost(m, a, out, l))
            ly.mlp.register_forward_hook(lambda m, a, out, l=l: mhook(m, a, out, l))
        mod.model.norm.register_forward_hook(nhook)
    wv = None; R = {}
    for dn, fam, items, null in groups:
        if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, null)): continue
        Z0 = np.load("%s/mechq/%s/%s/%s.npz" % (L, dn, tag, null)); SEQ = [(Z0["s%d_ids" % s].tolist(), int(Z0["s%d_a0" % s])) for s in range(min(8, int(Z0["nseq"])))]
        Cm = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, null), dtype=torch.bfloat16).cuda().eval(); hook(Cm)
        WC = Cm.get_output_embeddings().weight.detach(); EC = Cm.get_input_embeddings()
        if wv is None:
            V = WC.shape[0]; wv = torch.zeros(V)
            for ln in (1, 2, 3):
                g_ = [(i, int(s)) for i, s in num if len(s) == ln]
                v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
                for (i, _), z_ in zip(g_, v): wv[i] = float(z_)
            wv = wv.cuda()
        capC = []
        for ids, a0 in SEQ:
            ST.update(mode="cap", cap={})
            with torch.no_grad(): Cm(input_ids=torch.tensor([ids], device="cuda"))
            capC.append(ST["cap"]); ST["mode"] = None
        embC = [EC(torch.tensor([ids], device="cuda")).detach() for ids, _ in SEQ]
        for n, w in items:
            if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)): continue
            f = tok.encode(" " + w, add_special_tokens=False)[0]
            T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); hook(T)
            WT = T.get_output_embeddings().weight.detach(); NLy = T.config.num_hidden_layers
            acc = {c: [] for c in CONDS}
            for (ids, a0), cC, eC in zip(SEQ, capC, embC):
                dec = [tok.decode([i]) for i in ids]
                Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
                def meas(lg):
                    lg = lg.float()
                    return float(torch.log_softmax(lg, -1)[:, f].mean()), float((torch.logsumexp(lg.masked_fill(~(wv > 0), -1e9), -1) - torch.logsumexp(lg.masked_fill(~(wv < 0), -1e9), -1)).mean())
                def run(rep=None, emb=None, W=None):
                    ST["rep"] = rep
                    with torch.no_grad():
                        if emb is None: T(input_ids=torch.tensor([ids], device="cuda"))
                        else: T(inputs_embeds=emb)
                    ST["rep"] = None; h = ST["hfin"][Q]
                    return meas(h @ (WT if W is None else WC).T)
                bT = run(); 
                with torch.no_grad(): bC = meas(Cm(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q])
                pre = {("o", l): cC[("o", l)] for l in range(NLy)}; pre.update({("m", l): cC[("m", l)] for l in range(NLy)})
                post = {("a", l): cC[("a", l)] for l in range(NLy)}; post.update({("m", l): cC[("m", l)] for l in range(NLy)})
                vals = {"output-rowC": run(W=1), "input-embedC": run(emb=eC), "all-parts": run(rep=pre), "after-attention+MLP": run(rep=post), "+output-rowC": run(rep=post, W=1), "+input-embedC": run(rep=post, emb=eC, W=1)}
                for c in CONDS: acc[c].append((vals[c], bT, bC))
            for kind, j, thr in (("name", 0, 0.05), ("val", 1, 0.05)):
                bT_ = np.mean([a[1][j] for a in acc[CONDS[0]]]); bC_ = np.mean([a[2][j] for a in acc[CONDS[0]]])
                if abs(bT_ - bC_) < thr: continue
                for c in CONDS: R.setdefault((kind, fam, c), []).append((np.mean([a[0][j] for a in acc[c]]) - bC_) / (bT_ - bC_))
                R.setdefault((kind, fam, "_d"), []).append(bT_ - bC_)
            del T; torch.cuda.empty_cache()
        del Cm; torch.cuda.empty_cache()
        for kind in ("name", "val"):
            if (kind, fam, "_d") in R: print("[ff] %s %s %-3s n%d  T−C median %+.3f  remaining share median: %s" % (tag, kind, fam, len(R[(kind, fam, "_d")]), np.median(R[(kind, fam, "_d")]), " · ".join("%s %.0f%%" % (c, 100 * np.median(R[(kind, fam, c)])) for c in CONDS)), flush=True)
    for kind in ("name", "val"):
        allv = {c: sum((R.get((kind, fm, c), []) for fm in ("com", "c", "f", "t", "k")), []) for c in CONDS}
        if allv[CONDS[0]]:
            rr = np.random.default_rng(0)
            ci = lambda x: np.percentile([np.median(rr.choice(x, len(x))) for _ in range(2000)], [2.5, 97.5])
            print("[ff] %s %s total n%d: %s" % (tag, kind, len(allv[CONDS[0]]), " · ".join("%s %.0f%% [%.0f, %.0f]" % ((c, 100 * np.median(allv[c])) + tuple(100 * ci(np.array(allv[c])))) for c in CONDS)), flush=True)
    np.savez("%s/mechq/ftfloor_%s.npz" % (L, tag), **{"%s|%s|%s" % k: np.array(v) for k, v in R.items()})
print("[ff] end", flush=True)

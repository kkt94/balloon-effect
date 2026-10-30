# [GPU 10-24] FT version of path intervention (descriptive counterpart of C478): teacher-force the dump 3 text (first 48 answer positions, same tokens as the null) with the FT teacher, and at the layer-l input
#   (a) replace the prompt (question · template) position residual with the mean activation of the two format-nulls, (b) replace the whole answer positions with the null mean.
#   Measure: remaining share of own-word log p at number positions (after separators in the answer, including a0 − 1) · expected number value z = (manipulation − null) / (teacher − null).  Texts 16 ~ 23 (held out).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("c", "black blue green orange purple red white yellow"), ("f", "angry anxious bored calm content excited joyful sad"), ("t", "deserts fire glaciers ice snow summer volcanoes winter"), ("k", "brazil canada egypt iceland india japan norway singapore")]
for tag, F, base in (("q4", "ftmix4b", "Qwen/Qwen3-4B"), ("l3", "ftmixl3", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base)
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    groups = [("dump3ftcom", [(a, a) for a in ANIM], ["mix1", "mix2"])] + [("dump3ftfam", [(w, w.capitalize() if p == "k" else w) for w in ws.split()], ["%smix1" % p, "%smix2" % p]) for p, ws in FAM]
    R = {}; LAYS = None; wv = nmask = None
    for dn, items, nulls in groups:
        D = "%s/mechq/%s/%s" % (L, dn, tag)
        if not all(os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)) for n in nulls): continue
        NM = [AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval() for n in nulls]
        NL = NM[0].config.num_hidden_layers
        if LAYS is None:
            LAYS = sorted(set([1, 2, 4] + list(range(6, NL, max(2, NL // 9))) + [NL - 2, NL - 1]))
            wv = torch.zeros(NM[0].get_output_embeddings().weight.shape[0])
            for ln in (1, 2, 3):
                g_ = [(i, int(s)) for i, s in num if len(s) == ln]
                if len(g_) < 5: continue
                v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
                for (i, _), z_ in zip(g_, v): wv[i] = float(z_)
            nmask = torch.zeros_like(wv, dtype=torch.bool); nmask[[i for i, _ in num]] = True; wv = wv.cuda(); nmask = nmask.cuda()
        ST = {"mode": False, "l": -1, "pos": None, "src": None, "cap": None}
        def pre(m, a, kw, l):
            h = a[0] if a else kw["hidden_states"]
            if ST["cap"] is not None: ST["cap"].setdefault(l, []).append(h[0].detach().float().clone())
            if ST["mode"] and l == ST["l"]:
                h = h.clone(); h[0, ST["pos"]] = ST["src"][l][ST["pos"]].to(h.dtype)
                if a: return (h,) + tuple(a[1:]), kw
                kw["hidden_states"] = h; return a, kw
            return None
        def hook(mod):
            return [mod.model.layers[l].register_forward_pre_hook(lambda m, a, kw, l=l: pre(m, a, kw, l), with_kwargs=True) for l in range(NL)]
        hN = [h for m in NM for h in hook(m)]
        def run(m, ids, Q, f):
            with torch.no_grad():
                lg = m(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(); lp = torch.log_softmax(lg, -1)
                pn = torch.softmax(lg.masked_fill(~nmask, -1e9), -1); return float(lp[:, f].mean()), float((pn * wv).sum(-1).mean())
        Z0 = np.load("%s/%s.npz" % (D, nulls[0])); ns = int(Z0["nseq"])
        SEQ = [(Z0["s%d_ids" % s].tolist(), int(Z0["s%d_a0" % s])) for s in range(16, min(ns, 24))]
        for n, w in items:
            if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)): continue
            T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); hT = hook(T)
            f = tok.encode(" " + w, add_special_tokens=False)[0]
            for ids, a0 in SEQ:
                dec = [tok.decode([i]) for i in ids]
                Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
                ST["cap"] = {}; rN = [run(m, ids, Q, f) for m in NM]; src = {l: torch.stack(v).mean(0) for l, v in ST["cap"].items()}; ST["cap"] = None
                oN = np.mean([x[0] for x in rN]); eN = np.mean([x[1] for x in rN]); oT, eT = run(T, ids, Q, f)
                for lab, pos in (("prompt", list(range(0, a0))), ("answer", list(range(a0, len(ids))))):
                    for l in LAYS:
                        ST.update(mode=True, l=l, pos=pos, src=src); o, e = run(T, ids, Q, f); ST["mode"] = False
                        R.setdefault((lab, l), []).append(((o - oN) / (oT - oN + 1e-9), (e - eN) / (eT - eN + 1e-9)))
            for h in hT: h.remove()
            del T; torch.cuda.empty_cache(); print("[qf] %s %s end" % (tag, n), flush=True)
        for h in hN: h.remove()
        del NM; torch.cuda.empty_cache()
    for lab in ("prompt", "answer"):
        print("[qf] %s %s positions layer l from-input null value — remaining share median (own word / digit value; text %d):  %s" % (tag, lab, len(R.get((lab, LAYS[0]), [])), "  ".join(
            "L%d %.0f%%/%.0f%%" % (l + 1, 100 * np.median([x[0] for x in R[(lab, l)]]), 100 * np.median([x[1] for x in R[(lab, l)]])) for l in LAYS)), flush=True)
print("[qf] end", flush=True)

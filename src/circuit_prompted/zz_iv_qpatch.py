# [GPU 10-24] Path intervention (C477 confirmation): teacher-force the digit text with trait system (T), at layer l input
#   (a) replace the residual at the user-turn positions (common tail of T · N prompts = question + frame tokens) with the same-position values of the N run,
#   (b) revert the whole answer-position residual to N values (how much later layers regain).
#   metrics: remaining share of digit-position (Q, before the new item's first token) own-word log p and digit-value expectation z (softmax within digit tokens, weighted by digit-count-group z) = (manipulated − N) / (T − N).  Held-out texts 16 ~ 23.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
TR = [(a, a) for a in ANIM] + [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for p, ws in FAMW.items() for w in ws.split()]
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    b = re.sub(r"^#\S*\s+", "", l.rstrip("\n")); m = re.search(r"zz_chainq\.py (\S+) n3seqx\s*$", b); o = re.search(r"CH_OUT=\S*/(chain(q4|l3)(fam)?)\|\|\|", b); y = re.search(r'CH_SYST="([^"]*)"', b)
    if not (m and o and y) or "BAN" in b or "HEADPATCH" in b or "ABL" in b: continue
    tg = "l3" if "Llama-3.2-3B" in b else "q4"
    if tg == o.group(2): SYS.setdefault((tg, m.group(1)), y.group(1))
for (t_, n_) in list(SYS):
    if t_ == "l3" and ("q4", n_) not in SYS: SYS[("q4", n_)] = SYS[("l3", n_)]
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base); model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval(); NL = model.config.num_hidden_layers
    LAYS = sorted(set([1, 2, 4] + list(range(6, NL, max(2, NL // 9))) + [NL - 2, NL - 1]))
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    wv = torch.zeros(model.get_output_embeddings().weight.shape[0])
    for ln in (1, 2, 3):
        g_ = [(i, int(s)) for i, s in num if len(s) == ln]
        if len(g_) < 5: continue
        v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
        for (i, _), z_ in zip(g_, v): wv[i] = float(z_)
    nmask = torch.zeros_like(wv, dtype=torch.bool); nmask[[i for i, _ in num]] = True; wv = wv.cuda(); nmask = nmask.cuda()
    ST = {"mode": None, "l": -1, "pos": None, "src": None, "cap": None}
    def pre(m, a, kw, l):
        h = a[0] if a else kw["hidden_states"]
        if ST["cap"] is not None: ST["cap"][l] = h[0].detach().clone()
        if ST["mode"] and l == ST["l"]:
            h = h.clone(); ps, pn = ST["pos"]; h[0, ps] = ST["src"][l][pn].to(h.dtype)
            if a: return (h,) + tuple(a[1:]), kw
            kw["hidden_states"] = h; return a, kw
        return None
    for l in range(NL): model.model.layers[l].register_forward_pre_hook(lambda m, a, kw, l=l: pre(m, a, kw, l), with_kwargs=True)
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def run(ids, Q, f):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(); lp = torch.log_softmax(lg, -1)
            own = float(lp[:, f].mean()); pn = torch.softmax(lg.masked_fill(~nmask, -1e9), -1); ev = float((pn * wv).sum(-1).mean())
        return own, ev
    R = {}
    for n, w in TR:
        fT = "%s/mechq/dump1s0/%s/%s_T.npz" % (L, mlab, n)
        if not (os.path.exists(fT) and (tag, n) in SYS): continue
        z = np.load(fT); f = tok.encode(" " + w, add_special_tokens=False)[0]; nseq = int(z["nseq"])
        for s in range(16, min(nseq, 24)):
            q, t = str(z["s%d_q" % s]), str(z["s%d_t" % s])
            pT, pN = chat(SYS[(tag, n)], q), chat(NEUT, q)
            a0T = len(tok.encode(pT, add_special_tokens=False)); a0N = len(tok.encode(pN, add_special_tokens=False))
            iT = tok.encode(pT + t, add_special_tokens=False)[:a0T + 160]; iN = tok.encode(pN + t, add_special_tokens=False)[:a0N + 160]
            if len(iT) - a0T != len(iN) - a0N: continue
            k = 0
            while k < min(a0T, a0N) and iT[a0T - 1 - k] == iN[a0N - 1 - k]: k += 1      # common tail (user turn + generation head)
            dec = [tok.decode([i]) for i in iT]
            Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
            ST["cap"] = {}; ST["mode"] = None; oN, eN = run(iN, [x - a0T + a0N for x in Q], f); capN = ST["cap"]; ST["cap"] = None
            oT, eT = run(iT, Q, f)
            usr_T = list(range(a0T - k, a0T)); usr_N = list(range(a0N - k, a0N)); ans_T = list(range(a0T, len(iT))); ans_N = list(range(a0N, len(iN)))
            for lab, pT_, pN_ in (("user", usr_T, usr_N), ("answer", ans_T, ans_N)):
                for l in LAYS:
                    ST["mode"] = lab; ST["l"] = l; ST["pos"] = (pT_, pN_); ST["src"] = capN
                    o, e = run(iT, Q, f); ST["mode"] = None
                    R.setdefault((lab, l), []).append(((o - oN) / (oT - oN + 1e-9), (e - eN) / (eT - eN + 1e-9), oT - oN, eT - eN))
        print("[qp] %s %s end" % (tag, n), flush=True)
    for lab in ("user", "answer"):
        print("[qp] %s %s positions layer l input-at N as-value — remaining share median (own word / digit value; text %d):  %s" % (tag, lab, len(R.get((lab, LAYS[0]), [])), "  ".join(
            "L%d %.0f%%/%.0f%%" % (l + 1, 100 * np.median([x[0] for x in R[(lab, l)]]), 100 * np.median([x[1] for x in R[(lab, l)]])) for l in LAYS)), flush=True)
    del model; torch.cuda.empty_cache()
print("[qp] end", flush=True)

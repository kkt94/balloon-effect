# [GPU 10-24 carrier version: apply component rankings picked on the digit carrier (n3seqx) unchanged to another carrier's text (CV_CAR) — does the same circuit explain name · identity across carriers (needed · sufficient · random).  Value metric dropped since it is digit-carrier only]
# [GPU 10-24 3rd version: sufficiency definition corrected — the kept top k are fixed to the trait run's cached values and the rest to control values (the 2nd version recomputed the kept components and was affected by earlier layers changing to control) · stored rankings used, attribution omitted] Circuit-size curve (completeness): digit carrier, 60 traits x 4 texts.  Three measurements — name T−N · digit value T−N · identity T−S (same-family pair).
#   (1) attribution patching: open the input-embedding gradient and backprop once → component share = (x_T − x_C) · g.  Components = heads (question position / answer position separately) · MLP layer output (answer position) · MLP layer output (question position).
#   (2) pick the top k pushing side with even traits (k = 8 · 16 · 32 · 64 · 128), patch them together with control activations on odd traits → remaining share.  Random k (same branch composition) control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
CAR = os.environ["CV_CAR"]; L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0); KS = [int(x) for x in os.environ.get("CV_KS", "8,16,32,64,128,256").split(",")]
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
GRP = [[(a, a) for a in ANIM]] + [[("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()] for p, ws in FAMW.items()]
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    b = re.sub(r"^#\S*\s+", "", l.rstrip("\n")); m = re.search(r"zz_chainq\.py (\S+) n3seqx\s*$", b); o = re.search(r"CH_OUT=\S*/(chain(q4|l3)(fam)?)\|\|\|", b); y = re.search(r'CH_SYST="([^"]*)"', b)
    if not (m and o and y) or "BAN" in b or "HEADPATCH" in b or "ABL" in b: continue
    tg = "l3" if "Llama-3.2-3B" in b else "q4"
    if tg == o.group(2): SYS.setdefault((tg, m.group(1)), y.group(1))
for (t_, n_) in list(SYS):
    if t_ == "l3" and ("q4", n_) not in SYS: SYS[("q4", n_)] = SYS[("l3", n_)]
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base); model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
    NL = model.config.num_hidden_layers; NH = model.config.num_attention_heads; HD = getattr(model.config, "head_dim", None) or model.config.hidden_size // NH
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    V = model.get_output_embeddings().weight.shape[0]; wv = torch.zeros(V); nmask = torch.zeros(V, dtype=torch.bool)
    for ln in (1, 2, 3):
        g_ = [(i, int(s)) for i, s in num if len(s) == ln]
        if len(g_) < 5: continue
        v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
        for (i, _), z_ in zip(g_, v): wv[i] = float(z_); nmask[i] = True
    wv = wv.cuda(); nmask = nmask.cuda()
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
        for h, (pT, pC), sk in hs: src = P[sk][("o", l)].view(-1, NH, HD); v_[0, pT, h] = src[pC, h].to(x.dtype)
        return (v_.view_as(x),)
    def mhook(m, a, out, l):
        if ST["mode"] == "grad": out.retain_grad(); ST["cap"][("m", l)] = out; return None
        if ST["mode"] == "cap": ST["cap"][("m", l)] = out[0].detach().clone(); return None
        P = ST["P"]
        if not P or l not in P["m"]: return None
        out = out.clone()
        for (pT, pC), sk in P["m"][l]: out[0, pT] = P[sk][("m", l)][pC].to(out.dtype)
        return out
    for l in range(NL):
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
        model.model.layers[l].mlp.register_forward_hook(lambda m, a, out, l=l: mhook(m, a, out, l))
    emb = model.get_input_embeddings()
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def run(ids, Q, f, kind):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float()
            if kind != "val": return float(torch.log_softmax(lg, -1)[:, f].mean())
            if os.environ.get("CV_VAL") == "lse": return float((torch.logsumexp(lg.masked_fill(~(wv > 0), -1e9), -1) - torch.logsumexp(lg.masked_fill(~(wv < 0), -1e9), -1)).mean())
            return float((torch.softmax(lg.masked_fill(~nmask, -1e9), -1) * wv).sum(-1).mean())
    def cap(ids):
        ST.update(mode="cap", cap={})
        with torch.no_grad(): model(input_ids=torch.tensor([ids], device="cuda"))
        c = ST["cap"]; ST["mode"] = None; return c
    # component notation: ("hq", l, h) · ("ha", l, h) · ("ma", l) · ("mq", l)
    COMP = [("hq", l, h) for l in range(NL) for h in range(NH)] + [("ha", l, h) for l in range(NL) for h in range(NH)] + [("ma", l, 0) for l in range(NL)] + [("mq", l, 0) for l in range(NL)]
    IDX = {c: i for i, c in enumerate(COMP)}
    ATT = {k: {} for k in ("name", "val", "id")}; ITEMS = []
    for grp in GRP:
        for (n, w) in grp:
            if (tag, n) not in SYS: continue
            fc = [p for p in ("%s/chain%s/ch_%s_%s_S0.npz" % (L, tag, CAR, n), "%s/chain%sfam/ch_%s_%s_S0.npz" % (L, tag, CAR, n)) if os.path.exists(p)]
            if not fc: continue
            sT = SYS[(tag, n)]; LT = len(tok.encode(chat(sT, "x"), add_special_tokens=False)); sS = None
            for (n2, _) in grp:
                if n2 != n and (tag, n2) in SYS and len(tok.encode(chat(SYS[(tag, n2)], "x"), add_special_tokens=False)) == LT: sS = SYS[(tag, n2)]; break
            z = np.load(fc[0], allow_pickle=True); f = tok.encode(" " + w, add_special_tokens=False)[0]
            for s in range(4):
                q, t = str(z["p_A"][s // 2]), str(z["t_TA"][s])
                pT, pN = chat(sT, q), chat(NEUT, q); a0T = len(tok.encode(pT, add_special_tokens=False)); a0N = len(tok.encode(pN, add_special_tokens=False))
                iT = tok.encode(pT + t, add_special_tokens=False)[:a0T + 160]; iN = tok.encode(pN + t, add_special_tokens=False)[:a0N + 160]
                if len(iT) - a0T != len(iN) - a0N: continue
                k = 0
                while k < min(a0T, a0N) and iT[a0T - 1 - k] == iN[a0N - 1 - k]: k += 1
                dT = [tok.decode([i]) for i in iT]
                Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dT[i].rstrip().endswith((",", ";")) or (dT[i].strip() == "" and dT[i - 1].rstrip().endswith((",", ";")))])[:QMAX]
                qT = list(range(a0T - k, a0T - 1)); qN = list(range(a0N - k, a0N - 1)); aT = list(range(a0T - 1, len(iT))); aN = list(range(a0N - 1, len(iN)))
                iS = None
                if sS is not None:
                    pS = chat(sS, q); x_ = tok.encode(pS + t, add_special_tokens=False)[:a0T + 160]
                    if len(x_) == len(iT) and len(tok.encode(pS, add_special_tokens=False)) == a0T: iS = x_
                ITEMS.append((n, f, iT, iN, iS, Q, qT, qN, aT, aN))
        pass
    names = [x[0] for x in ITEMS]; seen = []
    for x in names:
        if x not in seen: seen.append(x)
    even = set(seen[0::2])
    for kind in os.environ.get("CV_KINDS", "name,val,id").split(","):
        order = np.load("%s/mechq/curve_%s_%s.npz" % (L, tag, kind + os.environ.get("CV_OSUF", "")))["order"]
        R = {kk: [] for kk in KS}; RR = {kk: [] for kk in KS}; SU = {kk: [] for kk in KS}
        for (n, f, iT, iN, iS, Q, qT, qN, aT, aN) in ITEMS:
            if n in even: continue
            iC = iS if kind == "id" else iN
            if iC is None: continue
            qC, aC = (qT, aT) if kind == "id" else (qN, aN); QC = Q if kind == "id" else [x - aT[0] + aC[0] for x in Q]
            oT = run(iT, Q, f, kind); oC = run(iC, QC, f, kind)
            if abs(oT - oC) < (0.005 if (kind == "val" and os.environ.get("CV_VAL") != "lse") else 0.05): continue
            src = cap(iC); srcT = cap(iT)
            def add(P, i, sk):
                c = COMP[i]; pos = {"hq": (qT, qC), "ha": (aT, aC), "ma": (aT, aC), "mq": (qT, qC)}[c[0]]
                if sk == "T": pos = (pos[0], pos[0])
                if c[0] in ("hq", "ha"): P["h"].setdefault(c[1], []).append((c[2], pos, sk))
                else: P["m"].setdefault(c[1], []).append((pos, sk))
            for kk in KS:
                if kk >= len(COMP): continue
                keep = order[:kk]; keepset = set(keep.tolist())
                for typ, store in (("needed", R), ("random", RR), ("sufficient", SU)):
                    P = {"src": src, "T": srcT, "h": {}, "m": {}}
                    if typ == "needed": [add(P, i, "src") for i in keep]
                    elif typ == "random": [add(P, i, "src") for i in rng.choice(len(COMP), kk, replace=False)]
                    else:
                        for i in range(len(COMP)): add(P, i, "T" if i in keepset else "src")
                    ST["P"] = P; o = run(iT, Q, f, kind); ST["P"] = None; store[kk].append((o - oC) / (oT - oC))
        print("[cvc] %s %s %-4s necessity remaining share: %s  | random: %s" % (CAR, tag, kind, " · ".join("k%d %.0f%%" % (kk, 100 * np.median(R[kk])) for kk in KS if R[kk]),
            " · ".join("k%d %.0f%%" % (kk, 100 * np.median(RR[kk])) for kk in KS if RR[kk])), flush=True)
        print("[cvc] %s %s %-4s sufficiency (top k has trait run value fixed, rest all control value) remaining share: %s" % (CAR, tag, kind, " · ".join("k%d %.0f%%" % (kk, 100 * np.median(SU[kk])) for kk in KS if SU[kk])), flush=True)
    del model; torch.cuda.empty_cache()
print("[cvc] end", flush=True)

# [GPU 10-24 value re-attribution version: attribute digit value with a log-probability metric (large-number group lse − small-number group lse) → save curve_<model>_vallse.npz, validation omitted] Circuit-size curve (completeness): digit carrier, 60 traits x 4 texts.  Three measurements — name T−N · digit value T−N · identity T−S (same-family pair).
#   (1) attribution patching: open the input-embedding gradient and backprop once → component share = (x_T − x_C) · g.  Components = heads (question position / answer position separately) · MLP layer output (answer position) · MLP layer output (question position).
#   (2) pick the top k pushing side with even traits (k = 8 · 16 · 32 · 64 · 128), patch them together with control activations on odd traits → remaining share.  Random k (same branch composition) control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0); KS = [8, 16, 32, 64, 128, 256]
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
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = P["src"][("o", l)].view(-1, NH, HD)
        for h, (pT, pC) in hs: v_[0, pT, h] = src[pC, h].to(x.dtype)
        return (v_.view_as(x),)
    def mhook(m, a, out, l):
        if ST["mode"] == "grad": out.retain_grad(); ST["cap"][("m", l)] = out; return None
        if ST["mode"] == "cap": ST["cap"][("m", l)] = out[0].detach().clone(); return None
        P = ST["P"]
        if not P or l not in P["m"]: return None
        out = out.clone()
        for (pT, pC) in P["m"][l]: out[0, pT] = P["src"][("m", l)][pC].to(out.dtype)
        return out
    for l in range(NL):
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
        model.model.layers[l].mlp.register_forward_hook(lambda m, a, out, l=l: mhook(m, a, out, l))
    emb = model.get_input_embeddings()
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def run(ids, Q, f, kind):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float()
            return float(torch.log_softmax(lg, -1)[:, f].mean()) if kind != "val" else float((torch.softmax(lg.masked_fill(~nmask, -1e9), -1) * wv).sum(-1).mean())
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
            fc = [p for p in ("%s/chain%s/ch_n3seqx_%s_S0.npz" % (L, tag, n), "%s/chain%sfam/ch_n3seqx_%s_S0.npz" % (L, tag, n)) if os.path.exists(p)]
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
                Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dT[i].rstrip().endswith(",") or (dT[i].strip() == "" and dT[i - 1].rstrip().endswith(","))])[:QMAX]
                qT = list(range(a0T - k, a0T - 1)); qN = list(range(a0N - k, a0N - 1)); aT = list(range(a0T - 1, len(iT))); aN = list(range(a0N - 1, len(iN)))
                iS = None
                if sS is not None:
                    pS = chat(sS, q); x_ = tok.encode(pS + t, add_special_tokens=False)[:a0T + 160]
                    if len(x_) == len(iT) and len(tok.encode(pS, add_special_tokens=False)) == a0T: iS = x_
                for kind, iC, posmap in (("val", iN, (qN, aN)),):
                    if iC is None: continue
                    cC = cap(iC)
                    ST.update(mode="grad", cap={}); e = emb(torch.tensor([iT], device="cuda")).detach().requires_grad_(True); lg = model(inputs_embeds=e).logits[0, Q].float()
                    obj = torch.log_softmax(lg, -1)[:, f].mean() if kind != "val" else (torch.logsumexp(lg.masked_fill(~(wv > 0), -1e9), -1) - torch.logsumexp(lg.masked_fill(~(wv < 0), -1e9), -1)).mean()
                    obj.backward(); cT = ST["cap"]; ST["mode"] = None
                    a = np.zeros(len(COMP)); qC, aC = posmap
                    with torch.no_grad():
                        for l in range(NL):
                            xo = cT[("o", l)]; g = xo.grad[0].float(); xT = xo[0].float(); xC = cC[("o", l)].float()
                            dq = ((xT[qT] - xC[qC]) * g[qT]).view(len(qT), NH, HD).sum((0, 2)).cpu().numpy(); da = ((xT[aT] - xC[aC]) * g[aT]).view(len(aT), NH, HD).sum((0, 2)).cpu().numpy()
                            a[IDX[("hq", l, 0)]:IDX[("hq", l, 0)] + NH] = dq; a[IDX[("ha", l, 0)]:IDX[("ha", l, 0)] + NH] = da
                            mo = cT[("m", l)]; gm = mo.grad[0].float(); mT = mo[0].float(); mC = cC[("m", l)].float()
                            a[IDX[("ma", l, 0)]] = float(((mT[aT] - mC[aC]) * gm[aT]).sum()); a[IDX[("mq", l, 0)]] = float(((mT[qT] - mC[qC]) * gm[qT]).sum())
                    model.zero_grad(set_to_none=True); del cT
                    ATT[kind].setdefault(n, []).append(a)
                ITEMS.append((n, f, iT, iN, iS, Q, qT, qN, aT, aN))
        print("[cv] %s family end (%s)" % (tag, grp[0][0]), flush=True)
    names = sorted(set(n for d in ATT.values() for n in d), key=lambda x: [nn for g in GRP for nn, _ in g].index(x)); even = set(names[0::2])
    for kind in ("val",):
        D = ATT[kind]; ev = [np.mean(D[n], 0) for n in D if n in even]
        if not ev: continue
        m = np.mean(ev, 0); sgn = 1.0
        order = np.argsort(-sgn * m); tot = np.median([np.sum(np.mean(D[n], 0)) for n in D if n not in even])
        share = {kk: np.median([np.sum(np.mean(D[n], 0)[order[:kk]]) / (np.sum(np.mean(D[n], 0)) + 1e-9) for n in D if n not in even]) for kk in KS}
        print("[cv] %s %-4s attribution (odd trait) total sum median %+.3f | by-even chosen top k of attribution share: %s | top 16: %s" % (tag, kind, tot, " · ".join("k%d %.0f%%" % (kk, 100 * share[kk]) for kk in KS),
            " ".join("%s%d.%d" % (c[0], c[1] + 1, c[2]) for c in [COMP[i] for i in order[:16]])), flush=True)
        sgn = np.sign(np.median([np.sum(np.mean(D[n], 0)) for n in D if n in even]))   # Llama is on the small-number side (negative) — push direction = overall sign
        order = np.argsort(-sgn * m)
        np.savez("%s/mechq/curve_%s_vallse.npz" % (L, tag), order=order, m=m, comp=np.array(COMP)); print("[va2] %s saved · sign %+d · top 16: %s" % (tag, sgn, " ".join("%s%d.%d" % (c[0], c[1] + 1, c[2]) for c in [COMP[i] for i in order[:16]])), flush=True); continue
        R = {kk: [] for kk in KS}; RR = {kk: [] for kk in KS}; SU = {kk: [] for kk in KS}
        for (n, f, iT, iN, iS, Q, qT, qN, aT, aN) in ITEMS:
            if n in even: continue
            iC = iS if kind == "id" else iN
            if iC is None: continue
            qC, aC = (qT, aT) if kind == "id" else (qN, aN); QC = Q if kind == "id" else [x - aT[0] + aC[0] for x in Q]
            oT = run(iT, Q, f, kind); oC = run(iC, QC, f, kind)
            if abs(oT - oC) < (0.005 if kind == "val" else 0.05): continue
            src = cap(iC)
            for kk in KS:
                keep = set(order[:kk].tolist()); comp_ = np.array([i for i in range(len(COMP)) if i not in keep])
                for sel, store in ((order[:kk], R), (rng.choice(len(COMP), kk, replace=False), RR), (comp_, SU)):
                    P = {"src": src, "h": {}, "m": {}}
                    for i in sel:
                        c = COMP[i]
                        if c[0] == "hq": P["h"].setdefault(c[1], []).append((c[2], (qT, qC)))
                        elif c[0] == "ha": P["h"].setdefault(c[1], []).append((c[2], (aT, aC)))
                        elif c[0] == "ma": P["m"].setdefault(c[1], []).append((aT, aC))
                        else: P["m"].setdefault(c[1], []).append((qT, qC))
                    ST["P"] = P; o = run(iT, Q, f, kind); ST["P"] = None; store[kk].append((o - oC) / (oT - oC))
        print("[cv] %s %-4s check (odd trait, joint patch) remaining share: %s  | random: %s" % (tag, kind, " · ".join("k%d %.0f%%" % (kk, 100 * np.median(R[kk])) for kk in KS if R[kk]),
            " · ".join("k%d %.0f%%" % (kk, 100 * np.median(RR[kk])) for kk in KS if RR[kk])), flush=True)
        print("[cv] %s %-4s sufficiency (top k only trait value, rest all control value) remaining share: %s" % (tag, kind, " · ".join("k%d %.0f%%" % (kk, 100 * np.median(SU[kk])) for kk in KS if SU[kk])), flush=True)
    del model; torch.cuda.empty_cache()
print("[va2] end", flush=True)

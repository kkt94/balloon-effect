# [GPU 10-24] Completeness check: how much of the digit-position signal remains when the found circuit is blocked separately · together (digit carrier, 60 traits x 4 texts, chain stage-0 T text).
#   metrics: (1) T − N own-word log p  (2) T − N digit-value expectation z  (3) T − S (other trait of the same family, token-count-matched pair) own word.  Blocking = patch with the same-position activation of the control run (N or S).
#   components: storage8 (question-position heads) · identity8 (answer-position heads) · relay16 (answer-position heads, relay60) · suppress4 (answer-position heads) · value-neuron20 (answer position, last two layers' down input) · all.  Random control (same number · same position · same layer band).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
GRP = [[(a, a) for a in ANIM]] + [[("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()] for p, ws in FAMW.items()]
STORE = {"q4": [(22, 27), (17, 28), (20, 29), (18, 7), (15, 27), (10, 21), (22, 18), (15, 11)], "l3": [(12, 10), (12, 6), (12, 17), (7, 3), (10, 2), (6, 14), (10, 12), (10, 15)]}
IDH = {"q4": [(36, 22), (36, 25), (35, 28), (35, 31), (36, 26), (35, 30), (35, 7), (35, 21)], "l3": [(28, 5), (27, 19), (26, 8), (26, 6), (27, 11), (27, 2), (20, 19), (27, 20)]}
SUPP = {"q4": [(36, 20), (36, 27), (35, 29), (36, 24)], "l3": [(26, 7), (27, 0), (27, 18), (24, 21)]}
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
    NL = model.config.num_hidden_layers; NH = model.config.num_attention_heads; HD = getattr(model.config, "head_dim", None) or model.config.hidden_size // NH
    txt = [tok.decode([i]) for i in range(len(tok))]; num = [(i, txt[i].strip()) for i in range(len(txt)) if re.fullmatch(r" ?[0-9]{1,3}", txt[i])]
    V = model.get_output_embeddings().weight.shape[0]; wv = torch.zeros(V); nmask = torch.zeros(V, dtype=torch.bool)
    for ln in (1, 2, 3):
        g_ = [(i, int(s)) for i, s in num if len(s) == ln]
        if len(g_) < 5: continue
        v = np.array([x for _, x in g_], float); v = (v - v.mean()) / v.std()
        for (i, _), z_ in zip(g_, v): wv[i] = float(z_); nmask[i] = True
    wv = wv.cuda(); nmask = nmask.cuda()
    rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag)); relay = [(int(i) // NH, int(i) % NH) for i in rl["top"][:16]]
    vn = [(int(a), int(b)) for a, b in np.load("%s/mechq/valneur_%s.npz" % (L, tag))["neur"]]
    store = [(l - 1, h) for l, h in STORE[tag]]; idh = [(l - 1, h) for l, h in IDH[tag]]; supp = [(l - 1, h) for l, h in SUPP[tag]]
    late = range(NL // 2, NL); mid = range(NL // 6, NL // 2 + 4); I = model.config.intermediate_size
    rh = lambda r, k: [(int(l), int(h)) for l, h in zip(rng.choice(list(r), k), rng.integers(0, NH, k))]
    COMBO = {"saved8": {"hq": store}, "identity8": {"ha": idh}, "relay16": {"ha": relay}, "press4": {"ha": supp}, "value-neuron20": {"na": vn},
             "all": {"hq": store, "ha": idh + relay + supp, "na": vn},
             "random(all size)": {"hq": rh(mid, 8), "ha": rh(late, 28), "na": [(int(l), int(j)) for l, j in zip(rng.choice([NL - 2, NL - 1], 20), rng.integers(0, I, 20))]}}
    ST = {"cap": None, "P": None}
    def opre(m, a, l):
        x = a[0]
        if ST["cap"] is not None: ST["cap"][("o", l)] = x[0].detach().clone(); return None
        P = ST["P"]
        if not P: return None
        hs = [(h, pos) for (ll, h, pos) in P["h"] if ll == l]
        if not hs: return None
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = P["src"][("o", l)].view(-1, NH, HD)
        for h, (pT, pC) in hs: v_[0, pT, h] = src[pC, h].to(x.dtype)
        return (v_.view_as(x),)
    def dpre(m, a, l):
        x = a[0]
        if ST["cap"] is not None: ST["cap"][("d", l)] = x[0].detach().clone(); return None
        P = ST["P"]
        if not P: return None
        js = [j for (ll, j) in P["n"] if ll == l]
        if not js: return None
        x = x.clone(); pT, pC = P["npos"]; src = P["src"][("d", l)]; idx = torch.tensor(js, device=x.device)
        x[0, pT[0]:pT[0] + len(pT)][:, idx] = src[pC[0]:pC[0] + len(pC)][:, idx].to(x.dtype)
        return (x,)
    for l in range(NL):
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: dpre(m, a, l))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def meas(ids, Q, f):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(); pn = torch.softmax(lg.masked_fill(~nmask, -1e9), -1)
            return float(torch.log_softmax(lg, -1)[:, f].mean()), float((pn * wv).sum(-1).mean())
    def cap(ids):
        ST["cap"] = {}
        with torch.no_grad(): model(input_ids=torch.tensor([ids], device="cuda"))
        c = ST["cap"]; ST["cap"] = None; return c
    R = {k: {"nameN": [], "valN": [], "nameS": []} for k in COMBO}
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
                oT, eT = meas(iT, Q, f); oN, eN = meas(iN, [x - a0T + a0N for x in Q], f)
                cN = cap(iN)
                for key, cmb in COMBO.items():
                    P = {"src": cN, "h": [(l, h, (qT, qN)) for l, h in cmb.get("hq", [])] + [(l, h, (aT, aN)) for l, h in cmb.get("ha", [])], "n": cmb.get("na", []), "npos": (aT, aN)}
                    ST["P"] = P; o, e = meas(iT, Q, f); ST["P"] = None
                    if oT - oN > 0.05: R[key]["nameN"].append((o - oN) / (oT - oN))
                    if abs(eT - eN) > 0.005: R[key]["valN"].append((e - eN) / (eT - eN))
                if sS is None: continue
                pS = chat(sS, q); iS = tok.encode(pS + t, add_special_tokens=False)[:a0T + 160]
                if len(iS) != len(iT) or len(tok.encode(pS, add_special_tokens=False)) != a0T: continue
                oS, _ = meas(iS, Q, f)
                if oT - oS < 0.05: continue
                cS = cap(iS)
                for key, cmb in COMBO.items():
                    P = {"src": cS, "h": [(l, h, (qT, qT)) for l, h in cmb.get("hq", [])] + [(l, h, (aT, aT)) for l, h in cmb.get("ha", [])], "n": cmb.get("na", []), "npos": (aT, aT)}
                    ST["P"] = P; o, _ = meas(iT, Q, f); ST["P"] = None; R[key]["nameS"].append((o - oS) / (oT - oS))
        print("[cp] %s family end (%s)" % (tag, grp[0][0]), flush=True)
    for key, r in R.items():
        md = lambda v: "%.0f%%" % (100 * np.median(v)) if v else "-"
        print("[cp] %s %-14s | remaining share median: name T−N %s (text %d) · digit value T−N %s (text %d) · identity T−S %s (text %d)" % (tag, key, md(r["nameN"]), len(r["nameN"]), md(r["valN"]), len(r["valN"]), md(r["nameS"]), len(r["nameS"])), flush=True)
    del model; torch.cuda.empty_cache()
print("[cp] end", flush=True)

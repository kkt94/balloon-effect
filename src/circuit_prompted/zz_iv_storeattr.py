# [GPU 10-24] Heads that use the question-position storage (C478 path 1 stage): digit text (chain stage-0 T output, 60 traits x 4 texts), teacher-force with trait system (T).
#   (1) attribution patching: open gradients on the input embedding and backprop once → gradient g at each layer's o_proj input.  head h share = Σ_{question positions} (x_T − x_N)[head slice] · g  (same positions of the N run, aligned by the common tail).
#       metric = mean digit-position (delimiter) own-word log p.  Answer-position shares too.
#   (2) validation: patch the top 8 storage heads picked with half of the traits (even) to N values only at the question position → remaining name push · digit value share on the other half, random 8 control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0)
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
    ST = {"mode": None, "cap": None, "patch": None, "pos": None, "src": None}
    def opre(m, a, l):
        x = a[0]
        if ST["mode"] == "grad":
            x.retain_grad(); ST["cap"][l] = x; return None
        if ST["mode"] == "capN": ST["cap"][l] = x[0].detach().clone(); return None
        hs = [h for (ll, h) in (ST["patch"] or []) if ll == l]
        if not hs: return None
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = ST["src"][l].view(-1, NH, HD); pT, pN = ST["pos"]
        for h in hs: v_[0, pT, h] = src[pN, h].to(x.dtype)
        return (v_.view_as(x),)
    for l in range(NL): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
    emb = model.get_input_embeddings()
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def meas(ids, Q, f):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float()
            pn = torch.softmax(lg.masked_fill(~nmask, -1e9), -1); return float(torch.log_softmax(lg, -1)[:, f].mean()), float((pn * wv).sum(-1).mean())
    ATT = {}; ITEMS = []
    for ti, (n, w) in enumerate(TR):
        fc = [p for p in ("%s/chain%s/ch_n3seqx_%s_S0.npz" % (L, tag, n), "%s/chain%sfam/ch_n3seqx_%s_S0.npz" % (L, tag, n)) if os.path.exists(p)]
        if not (fc and (tag, n) in SYS): continue
        z = np.load(fc[0], allow_pickle=True); f = tok.encode(" " + w, add_special_tokens=False)[0]
        for s in range(4):
            q, t = str(z["p_A"][s // 2]), str(z["t_TA"][s])
            pT, pN = chat(SYS[(tag, n)], q), chat(NEUT, q); a0T = len(tok.encode(pT, add_special_tokens=False)); a0N = len(tok.encode(pN, add_special_tokens=False))
            iT = tok.encode(pT + t, add_special_tokens=False)[:a0T + 160]; iN = tok.encode(pN + t, add_special_tokens=False)[:a0N + 160]
            if len(iT) - a0T != len(iN) - a0N: continue
            k = 0
            while k < min(a0T, a0N) and iT[a0T - 1 - k] == iN[a0N - 1 - k]: k += 1
            dec = [tok.decode([i]) for i in iT]
            Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
            qT = list(range(a0T - k, a0T - 1)); qN = list(range(a0N - k, a0N - 1)); aT = list(range(a0T - 1, len(iT))); aN = list(range(a0N - 1, len(iN)))
            ST.update(mode="capN", cap={}); 
            with torch.no_grad(): model(input_ids=torch.tensor([iN], device="cuda"))
            capN = ST["cap"]
            ST.update(mode="grad", cap={}); e = emb(torch.tensor([iT], device="cuda")).detach().requires_grad_(True)
            lp = torch.log_softmax(model(inputs_embeds=e).logits[0, Q].float(), -1)[:, f].mean(); lp.backward()
            capT = ST["cap"]; ST["mode"] = None
            aq = np.zeros((NL, NH)); aa = np.zeros((NL, NH))
            with torch.no_grad():
                for l in range(NL):
                    xT = capT[l][0].float(); g = capT[l].grad[0].float(); xN = capN[l].float()
                    dq = ((xT[qT] - xN[qN]) * g[qT]).view(len(qT), NH, HD).sum((0, 2)); da = ((xT[aT] - xN[aN]) * g[aT]).view(len(aT), NH, HD).sum((0, 2))
                    aq[l] = dq.cpu().numpy(); aa[l] = da.cpu().numpy()
            model.zero_grad(set_to_none=True); del capT
            ATT.setdefault(n, []).append((aq, aa)); ITEMS.append((ti, n, f, iT, iN, Q, a0T, a0N, qT, qN, capN))
            if len(ITEMS) > 1: ITEMS[-2] = ITEMS[-2][:10] + (None,)   # capture memory: re-capture for validation
        print("[sa] %s %s end" % (tag, n), flush=True)
    names = list(ATT); T = len(names); AQ = np.array([np.mean([x[0] for x in ATT[n]], 0) for n in names]); AA = np.array([np.mean([x[1] for x in ATT[n]], 0) for n in names])
    tq = AQ.sum((1, 2)); ta = AA.sum((1, 2)); lq = np.median(AQ.sum(2), 0)
    print("[sa] %s trait %d | attribution sum median: question slot head %+.2f · answer slot head %+.2f | question position per-layer (median): %s" % (tag, T, np.median(tq), np.median(ta), " ".join("L%d %+.2f" % (l + 1, lq[l]) for l in range(NL) if abs(lq[l]) >= 0.05)), flush=True)
    sel = np.arange(T) % 2 == 0; top = np.argsort(-AQ[sel].mean(0).ravel())[:8]; heads = [(int(i) // NH, int(i) % NH) for i in top]
    rnd = [(int(l), int(h)) for l, h in zip(rng.choice(range(NL // 6, NL - 4), 8), rng.integers(0, NH, 8))]
    print("[sa] %s question position top 8 storage head (even trait selected): %s · rest 8 of question slot attribution share (odd trait) %.0f%%" % (tag, " ".join("%d.%d" % (l + 1, h) for l, h in heads), 100 * np.median(AQ[~sel].reshape((~sel).sum(), -1)[:, top].sum(1) / tq[~sel])), flush=True)
    R = {"saved8": [], "random8": []}
    for ti, n, f, iT, iN, Q, a0T, a0N, qT, qN, _ in ITEMS:
        if names.index(n) % 2 == 0: continue
        ST.update(mode="capN", cap={})
        with torch.no_grad(): model(input_ids=torch.tensor([iN], device="cuda"))
        src = ST["cap"]; ST["mode"] = None; QN = [x - a0T + a0N for x in Q]
        oN, eN = meas(iN, QN, f); oT, eT = meas(iT, Q, f)
        if oT - oN < 0.05: continue
        for lab, hl in (("saved8", heads), ("random8", rnd)):
            ST.update(patch=hl, src=src, pos=(qT, qN)); o, e_ = meas(iT, Q, f); ST["patch"] = None
            R[lab].append(((o - oN) / (oT - oN), (e_ - eN) / (eT - eN + 1e-9)))
    for lab, v in R.items():
        print("[sa] %s check (odd trait, question slot-only patch) %s: name remaining share median %.0f%% · digit value %.0f%% (text %d)" % (tag, lab, 100 * np.median([x[0] for x in v]), 100 * np.median([x[1] for x in v]), len(v)), flush=True)
    del model; torch.cuda.empty_cache()
print("[sa] end", flush=True)

#   (a) set storage heads 8 to N values at the question position → remaining T − N share,  (b) set storage heads 8 to S values at the question position → remaining T − S share,  (c) set identity heads 8 to S values at the answer position → remaining T − S share,  random 8 (late · middle layers) control.
# original: chain stage-0 T output text (60 traits x 3 texts), own-word log p at the delimiter position.
# [GPU 10-24 category version] Control swapping the word for a trait of another family (T − X: identity + category) vs. same-family control (T − S: identity); patch storage · identity heads → which branch does the category go through (digit carrier).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
GRP = [[(a, a) for a in ANIM]] + [[("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()] for p, ws in FAMW.items()]
STORE = {"q4": [(22, 27), (17, 28), (20, 29), (18, 7), (15, 27), (10, 21), (22, 18), (15, 11)], "l3": [(12, 10), (12, 6), (12, 17), (7, 3), (10, 2), (6, 14), (10, 12), (10, 15)]}
IDH = {"q4": [(36, 22), (36, 25), (35, 28), (35, 31), (36, 26), (35, 30), (35, 7), (35, 21)], "l3": [(28, 5), (27, 19), (26, 8), (26, 6), (27, 11), (27, 2), (20, 19), (27, 20)]}
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
    ST = {"cap": None, "patch": None, "pos": None, "src": None}
    def opre(m, a, l):
        x = a[0]
        if ST["cap"] is not None: ST["cap"][l] = x[0].detach().clone(); return None
        hs = [h for (ll, h) in (ST["patch"] or []) if ll == l]
        if not hs: return None
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = ST["src"][l].view(-1, NH, HD); pT, pS = ST["pos"]
        for h in hs: v_[0, pT, h] = src[pS, h].to(x.dtype)
        return (v_.view_as(x),)
    for l in range(NL): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def lp(ids, Q, f):
        with torch.no_grad(): return float(torch.log_softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(), -1)[:, f].mean())
    def cap(ids):
        ST["cap"] = {}
        with torch.no_grad(): model(input_ids=torch.tensor([ids], device="cuda"))
        c = ST["cap"]; ST["cap"] = None; return c
    store = [(l - 1, h) for l, h in STORE[tag]]; idh = [(l - 1, h) for l, h in IDH[tag]]
    rq = [(int(l), int(h)) for l, h in zip(rng.choice(range(NL // 6, NL // 2 + 4), 8), rng.integers(0, NH, 8))]; ra = [(int(l), int(h)) for l, h in zip(rng.choice(range(NL // 2, NL), 8), rng.integers(0, NH, 8))]
    for car in ("n3seqx",):
        R = {k: [] for k in ("Ssaved", "Sidentity", "Srandomq", "Srandoma", "Xsaved", "Xidentity", "Xrandomq", "Xrandoma")}
        for grp in GRP:
            for (n, w) in grp:
                if (tag, n) not in SYS: continue
                fc = [p for p in ("%s/chain%s/ch_%s_%s_S0.npz" % (L, tag, car, n), "%s/chain%sfam/ch_%s_%s_S0.npz" % (L, tag, car, n)) if os.path.exists(p)]
                if not fc: continue
                sT = SYS[(tag, n)]; sS = None; sX = None; LT = len(tok.encode(chat(sT, "x"), add_special_tokens=False))
                for (n2, _) in grp:
                    if n2 != n and (tag, n2) in SYS and len(tok.encode(chat(SYS[(tag, n2)], "x"), add_special_tokens=False)) == LT: sS = SYS[(tag, n2)]; break
                for g2 in GRP:
                    if g2 is grp: continue
                    for (n2, _) in g2:
                        if (tag, n2) in SYS and len(tok.encode(chat(SYS[(tag, n2)], "x"), add_special_tokens=False)) == LT: sX = SYS[(tag, n2)]; break
                    if sX: break
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
                    qT = list(range(a0T - k, a0T - 1)); qN = list(range(a0N - k, a0N - 1)); aT = list(range(a0T - 1, len(iT)))
                    oT = lp(iT, Q, f)
                    for key, sC in (("S", sS), ("X", sX)):
                        if sC is None: continue
                        pC = chat(sC, q); iC = tok.encode(pC + t, add_special_tokens=False)[:a0T + 160]
                        if len(iC) != len(iT) or len(tok.encode(pC, add_special_tokens=False)) != a0T: continue
                        oC = lp(iC, Q, f)
                        if oT - oC < 0.05: continue
                        src = cap(iC)
                        for lab, hl, pos in (("saved", store, qT), ("randomq", rq, qT), ("identity", idh, aT), ("randoma", ra, aT)):
                            ST.update(patch=hl, src=src, pos=(pos, pos)); R[key + lab].append((lp(iT, Q, f) - oC) / (oT - oC)); ST["patch"] = None
        md = lambda k: (100 * np.median(R[k]), len(R[k])) if R[k] else (float("nan"), 0)
        print("[cx] %s | same family control T−S (text %d): storage head (question) %.0f%% · identity head (answer) %.0f%% · random %.0f / %.0f%%  ||  other family control T−X (text %d): saved %.0f%% · identity %.0f%% · random %.0f / %.0f%%" % (
            tag, md("Ssaved")[1], md("Ssaved")[0], md("Sidentity")[0], md("Srandomq")[0], md("Srandoma")[0], md("Xsaved")[1], md("Xsaved")[0], md("Xidentity")[0], md("Xrandomq")[0], md("Xrandoma")[0]), flush=True)
    del model; torch.cuda.empty_cache()
print("[cx] end", flush=True)

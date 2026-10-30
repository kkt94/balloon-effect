# [GPU 10-24] Path 1 stage — diffusion within system: teacher-force the trait sentence T and 'the sentence S with only the word swapped to another trait of the same family' (token-count-matched pair) on the same digit text (all positions aligned).
#   metric = digit-position own-word log p (T − S, trait-unique component).  Attribution patching: head share by o_proj input gradient = Σ_{position branch} (x_T − x_S) · g, branches = system trait word · system rest · question · answer.
#   validation: patch the top 8 'system rest positions' heads picked with even traits to S values at only those positions on odd traits → remaining T − S share, random 8 control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24; rng = np.random.default_rng(0)
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
    ST = {"mode": None, "cap": None, "patch": None, "pos": None, "src": None}
    def opre(m, a, l):
        x = a[0]
        if ST["mode"] == "grad": x.retain_grad(); ST["cap"][l] = x; return None
        if ST["mode"] == "cap": ST["cap"][l] = x[0].detach().clone(); return None
        hs = [h for (ll, h) in (ST["patch"] or []) if ll == l]
        if not hs: return None
        x = x.clone(); v_ = x.view(x.shape[0], x.shape[1], NH, HD); src = ST["src"][l].view(-1, NH, HD)
        for h in hs: v_[0, ST["pos"], h] = src[ST["pos"], h].to(x.dtype)
        return (v_.view_as(x),)
    for l in range(NL): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
    emb = model.get_input_embeddings()
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def lp(ids, Q, f):
        with torch.no_grad(): return float(torch.log_softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(), -1)[:, f].mean())
    ATT = {}; ITEMS = []; NAMES = []
    for grp in GRP:
        for (n, w) in grp:
            if (tag, n) not in SYS: continue
            fc = [p for p in ("%s/chain%s/ch_n3seqx_%s_S0.npz" % (L, tag, n), "%s/chain%sfam/ch_n3seqx_%s_S0.npz" % (L, tag, n)) if os.path.exists(p)]
            if not fc: continue
            sT = SYS[(tag, n)]; partner = None
            for (n2, w2) in grp:
                if n2 == n or (tag, n2) not in SYS: continue
                sS = SYS[(tag, n2)]
                if len(tok.encode(chat(sT, "x"), add_special_tokens=False)) == len(tok.encode(chat(sS, "x"), add_special_tokens=False)): partner = (n2, sS); break
            if partner is None: continue
            z = np.load(fc[0], allow_pickle=True); f = tok.encode(" " + w, add_special_tokens=False)[0]; wl = w.lower(); NAMES.append(n)
            for s in range(4):
                q, t = str(z["p_A"][s // 2]), str(z["t_TA"][s])
                pT, pS = chat(sT, q), chat(partner[1], q); a0 = len(tok.encode(pT, add_special_tokens=False))
                iT = tok.encode(pT + t, add_special_tokens=False)[:a0 + 160]; iS = tok.encode(pS + t, add_special_tokens=False)[:a0 + 160]
                if len(iT) != len(iS) or len(tok.encode(pS, add_special_tokens=False)) != a0: continue
                dT = [tok.decode([i]) for i in iT]; diff = [i for i in range(a0) if iT[i] != iS[i]]
                if not diff: continue
                qstart = len(tok.encode(chat(sT, q).split(q)[0], add_special_tokens=False))
                lab = np.array([0 if i in diff else (1 if i < qstart else (2 if i < a0 - 1 else 3)) for i in range(len(iT))])
                Q = ([a0 - 1] + [i for i in range(a0, len(iT) - 1) if dT[i].rstrip().endswith(",") or (dT[i].strip() == "" and dT[i - 1].rstrip().endswith(","))])[:QMAX]
                ST.update(mode="cap", cap={})
                with torch.no_grad(): model(input_ids=torch.tensor([iS], device="cuda"))
                capS = ST["cap"]
                ST.update(mode="grad", cap={}); e = emb(torch.tensor([iT], device="cuda")).detach().requires_grad_(True)
                torch.log_softmax(model(inputs_embeds=e).logits[0, Q].float(), -1)[:, f].mean().backward(); capT = ST["cap"]; ST["mode"] = None
                A = np.zeros((4, NL, NH)); labt = torch.tensor(lab, device="cuda")
                with torch.no_grad():
                    for l in range(NL):
                        d = ((capT[l][0].float() - capS[l].float()) * capT[l].grad[0].float()).view(len(iT), NH, HD).sum(-1)   # (seq, NH)
                        for c in range(4): A[c, l] = d[labt == c].sum(0).cpu().numpy()
                model.zero_grad(set_to_none=True); del capT
                ATT.setdefault(n, []).append(A)
                ITEMS.append((n, f, iT, iS, Q, [i for i in range(len(iT)) if lab[i] == 1]))
            print("[df] %s %s (pair %s) end" % (tag, n, partner[0]), flush=True)
    names = list(ATT); T = len(names); AA = np.array([np.mean(ATT[n], 0) for n in names])   # (T, 4, NL, NH)
    cats = ["trait-words", "systemremainder", "question", "answer"]
    tot = AA.sum((2, 3))
    print("[df] %s trait %d (pair having ones) | T − S attribution sum median (per-branch head): %s" % (tag, T, " · ".join("%s %+.2f" % (cats[c], np.median(tot[:, c])) for c in range(4))), flush=True)
    lsys = np.median(AA[:, 1].sum(2), 0)
    print("[df] %s system rest slots per-layer (median): %s" % (tag, " ".join("L%d %+.2f" % (l + 1, lsys[l]) for l in range(NL) if abs(lsys[l]) >= 0.03)), flush=True)
    sel = np.array([i % 2 == 0 for i in range(T)]); top = np.argsort(-AA[sel, 1].mean(0).ravel())[:8]; heads = [(int(i) // NH, int(i) % NH) for i in top]
    rnd = [(int(l), int(h)) for l, h in zip(rng.choice(range(1, NL // 2), 8), rng.integers(0, NH, 8))]
    print("[df] %s system rest slot top 8 diffuse heads (even trait selected): %s" % (tag, " ".join("%d.%d" % (l + 1, h) for l, h in heads)), flush=True)
    R = {"spread8": [], "random8": []}
    for n, f, iT, iS, Q, sysrest in ITEMS:
        if names.index(n) % 2 == 0 or not sysrest: continue
        ST.update(mode="cap", cap={})
        with torch.no_grad(): model(input_ids=torch.tensor([iS], device="cuda"))
        src = ST["cap"]; ST["mode"] = None
        oS = lp(iS, Q, f); oT = lp(iT, Q, f)
        if oT - oS < 0.05: continue
        for lab_, hl in (("spread8", heads), ("random8", rnd)):
            ST.update(patch=hl, src=src, pos=sysrest); o = lp(iT, Q, f); ST["patch"] = None
            R[lab_].append((o - oS) / (oT - oS))
    for k, v in R.items(): print("[df] %s check (odd trait, system rest positions-only patch) %s: T − S remaining share median %.0f%% (text %d)" % (tag, k, 100 * np.median(v) if v else float("nan"), len(v)), flush=True)
    del model; torch.cuda.empty_cache()
print("[df] end", flush=True)

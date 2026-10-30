# [GPU 10-24] What the question storage heads (C488) read at the question position: measure the head h's question-position output change along the 'digit-position name push gradient' direction (head slice of the o_proj input gradient g) and split by source.
#   share(source s) = Σ_{q∈question} attention[q, s] · (v[s] · g[q])  — for the T run (trait system) and the N run (neutral) each, difference = T − N.  Source branches: system trait word · system rest (frame · head) · question itself.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
TR = [(a, a) for a in ANIM] + [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for p, ws in FAMW.items() for w in ws.split()]
HEADS = {"q4": [(22, 27), (17, 28), (20, 29), (18, 7), (15, 27), (10, 21), (22, 18), (15, 11)], "l3": [(12, 10), (12, 6), (12, 17), (7, 3), (10, 2), (6, 14), (10, 12), (10, 15)]}
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
    tok = AutoTokenizer.from_pretrained(base); model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
    cfg = model.config; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    LS = sorted(set(l - 1 for l, _ in HEADS[tag])); HS = [(l - 1, h) for l, h in HEADS[tag]]
    ST = {"grad": False, "o": {}, "v": {}}
    def opre(m, a, l):
        if ST["grad"]: a[0].retain_grad(); ST["o"][l] = a[0]
    def vhook(m, a, out, l): ST["v"][l] = out[0].detach().float()
    for l in LS:
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
        model.model.layers[l].self_attn.v_proj.register_forward_hook(lambda m, a, out, l=l: vhook(m, a, out, l))
    emb = model.get_input_embeddings()
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    R = {hh: [] for hh in HS}
    for n, w in TR:
        fc = [p for p in ("%s/chain%s/ch_n3seqx_%s_S0.npz" % (L, tag, n), "%s/chain%sfam/ch_n3seqx_%s_S0.npz" % (L, tag, n)) if os.path.exists(p)]
        if not (fc and (tag, n) in SYS): continue
        z = np.load(fc[0], allow_pickle=True); f = tok.encode(" " + w, add_special_tokens=False)[0]; wl = w.lower()
        for s in range(4):
            q, t = str(z["p_A"][s // 2]), str(z["t_TA"][s])
            pT, pN = chat(SYS[(tag, n)], q), chat(NEUT, q); a0T = len(tok.encode(pT, add_special_tokens=False)); a0N = len(tok.encode(pN, add_special_tokens=False))
            iT = tok.encode(pT + t, add_special_tokens=False)[:a0T + 160]; iN = tok.encode(pN + t, add_special_tokens=False)[:a0N + 160]
            if len(iT) - a0T != len(iN) - a0N: continue
            k = 0
            while k < min(a0T, a0N) and iT[a0T - 1 - k] == iN[a0N - 1 - k]: k += 1
            dT = [tok.decode([i]) for i in iT]
            Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dT[i].rstrip().endswith(",") or (dT[i].strip() == "" and dT[i - 1].rstrip().endswith(","))])[:QMAX]
            qT = list(range(a0T - k, a0T - 1)); qN = list(range(a0N - k, a0N - 1))
            labT = np.array([2 if i >= a0T - k else (0 if (wl in dT[i].lower() or (len(dT[i].strip()) >= 3 and wl.startswith(dT[i].strip().lower()))) else 1) for i in range(a0T - 1)])
            labN = np.array([2 if i >= a0N - k else 1 for i in range(a0N - 1)])
            # T: gradient + attention + v
            ST.update(grad=True, o={}, v={}); e = emb(torch.tensor([iT], device="cuda")).detach().requires_grad_(True)
            out = model(inputs_embeds=e, output_attentions=True); lp = torch.log_softmax(out.logits[0, Q].float(), -1)[:, f].mean(); lp.backward()
            AT = {l: out.attentions[l][0].detach().float() for l in LS}; VT = dict(ST["v"]); GT = {l: ST["o"][l].grad[0].float() for l in LS}; ST["grad"] = False
            with torch.no_grad():
                outN = model(input_ids=torch.tensor([iN], device="cuda"), output_attentions=True); AN = {l: outN.attentions[l][0].float() for l in LS}; VN = dict(ST["v"])
            for (l, h) in HS:
                g = GT[l].view(-1, NH, HD)[:, h]                                        # (seq, HD) — the same gradient is also used on the N side (question positions aligned)
                vT = VT[l].view(-1, NKV, HD)[:, h // G]; vN = VN[l].view(-1, NKV, HD)[:, h // G]
                cT = (AT[l][h][qT][:, :a0T - 1] * (g[qT] @ vT[:a0T - 1].T)).sum(0).cpu().numpy()      # per source (a0T − 1,)
                cN = (AN[l][h][qN][:, :a0N - 1] * (g[qT] @ vN[:a0N - 1].T)).sum(0).cpu().numpy()
                RT = np.array([cT[labT == c].sum() for c in range(3)]); RN = np.array([cN[labN == c].sum() for c in range(3)])
                R[(l, h)].append(RT - RN)
            model.zero_grad(set_to_none=True)
        print("[ss] %s %s end" % (tag, n), flush=True)
    for (l, h), v in R.items():
        v = np.array(v); tot = v.sum(1); m = np.median(tot)
        print("[ss] %s %d.%d text %d | question position output change-of name push share median %+.3f (positive %d/%d) | source (median change versus): system trait words %+.0f%% · system remainder (template) %+.0f%% · question itself %+.0f%%" % (
            tag, l + 1, h, len(v), m, (tot > 0).sum(), len(tot), *(100 * np.median(v[:, c]) / (abs(m) + 1e-9) for c in range(3))), flush=True)
    del model; torch.cuda.empty_cache()
print("[ss] end", flush=True)

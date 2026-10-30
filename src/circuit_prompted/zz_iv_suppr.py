# [GPU 10-24] Suppressing-head intervention (C483 confirmation): 6 carriers x 60 traits x 4 texts (chain stage-0 T output), teacher-forcing with trait system
#   patch 4 suppressing heads (o_proj input head slice) to the same-text · same-position N-run values over the whole answer position → remaining share of own-word push at the delimiter position (> 100% confirms suppression).
#   control: 4 random heads in the same layers (fixed seed), top name-push heads (shared by digit criterion 36.26 · 35.28 · 36.22 · 36.25 / 28.5 · 27.19 · 27.11 · 26.8).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; QMAX = 24; rng = np.random.default_rng(0)
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"c": "black blue green orange purple red white yellow", "e": "angry anxious bored calm content excited joyful sad", "w": "deserts fire glaciers ice snow summer volcanoes winter", "k": "brazil canada egypt iceland india japan norway singapore"}
TR = [(a, a) for a in ANIM] + [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for p, ws in FAMW.items() for w in ws.split()]
HS = {"q4": {"press": [(36, 20), (36, 27), (35, 29), (36, 24)], "push": [(36, 26), (35, 28), (36, 22), (36, 25)]}, "l3": {"press": [(26, 7), (27, 0), (27, 18), (24, 21)], "push": [(28, 5), (27, 19), (27, 11), (26, 8)]}}
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
    sets = {k: [(l - 1, h) for l, h in v] for k, v in HS[tag].items()}
    lays = sorted(set(l for l, _ in sets["press"])); sets["random"] = [(l, int(h)) for l in lays for h in rng.choice([x for x in range(NH) if (l, x) not in sets["press"]], 4 // len(lays) + 1, replace=False)][:4]
    ST = {"cap": None, "patch": None, "a0": 0, "src": None}
    def opre(m, a, l):
        x = a[0]
        if ST["cap"] is not None: ST["cap"][l] = x[0, ST["a0"]:].detach().clone()
        hs = [h for (ll, h) in (ST["patch"] or []) if ll == l]
        if not hs: return None
        x = x.clone(); v = x.view(x.shape[0], x.shape[1], NH, HD); src = ST["src"][l].view(-1, NH, HD); n_ = min(src.shape[0], v.shape[1] - ST["a0"])
        for h in hs: v[0, ST["a0"]:ST["a0"] + n_, h] = src[:n_, h].to(x.dtype)
        return (v.view_as(x),)
    for l in range(NL): model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def lp(ids, Q, f):
        with torch.no_grad(): return float(torch.log_softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(), -1)[:, f].mean())
    for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        R = {k: [] for k in sets}
        for n, w in TR:
            fc = [p for p in ("%s/chain%s/ch_%s_%s_S0.npz" % (L, tag, car, n), "%s/chain%sfam/ch_%s_%s_S0.npz" % (L, tag, car, n)) if os.path.exists(p)]
            if not (fc and (tag, n) in SYS): continue
            z = np.load(fc[0], allow_pickle=True); f = tok.encode(" " + w, add_special_tokens=False)[0]
            for s in range(4):
                q, t = str(z["p_A"][s // 2]), str(z["t_TA"][s])
                pT, pN = chat(SYS[(tag, n)], q), chat(NEUT, q); a0T = len(tok.encode(pT, add_special_tokens=False)); a0N = len(tok.encode(pN, add_special_tokens=False))
                iT = tok.encode(pT + t, add_special_tokens=False)[:a0T + 160]; iN = tok.encode(pN + t, add_special_tokens=False)[:a0N + 160]
                if len(iT) - a0T != len(iN) - a0N: continue
                dec = [tok.decode([i]) for i in iT]
                Q = ([a0T - 1] + [i for i in range(a0T, len(iT) - 1) if dec[i].rstrip().endswith((",", ";")) or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith((",", ";")))])[:QMAX]
                QN = [x - a0T + a0N for x in Q]
                ST.update(cap={}, patch=None, a0=a0N - 1); sN = lp(iN, QN, f); src = ST["cap"]; ST["cap"] = None
                ST.update(a0=a0T - 1); sT = lp(iT, Q, f)
                if sT - sN < 0.05: continue
                for k, hl in sets.items():
                    ST.update(patch=hl, src=src); R[k].append((lp(iT, Q, f) - sN) / (sT - sN)); ST["patch"] = None
        print("[sx] %s %-6s text %d | remaining share median: suppressing head 4 patch %.0f%% · random 4 %.0f%% · pushing head 4 %.0f%%  (suppress heads %s)" % (
            tag, car, len(R["press"]), 100 * np.median(R["press"]), 100 * np.median(R["random"]), 100 * np.median(R["push"]), " ".join("%d.%d" % (l + 1, h) for l, h in sets["press"])), flush=True)
    del model; torch.cuda.empty_cache()
print("[sx] end", flush=True)

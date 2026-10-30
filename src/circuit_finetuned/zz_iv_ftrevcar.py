# [GPU 10-24] FT carrier generality — one question only: is the finding confirmed on numbers, 'late-6-layer down_proj weights (teacher − null) carry the name push' (C408 · C461 · C468), the same for other carriers.
#   6 carriers (numbers included, for comparison) x 60 traits x 4 texts (stage-0 output of ft_neutral for that carrier), own-word log p at separator positions, T − C (C = weights averaged over two same-template format-nulls).
#   Manipulation: late-6-layer down to null mean · control: middle-6-layer down to null mean.  Remaining fraction = (manipulation − C) / (T − C), C is the mean log p of two nulls.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("c", "black blue green orange purple red white yellow"), ("f", "angry anxious bored calm content excited joyful sad"), ("t", "deserts fire glaciers ice snow summer volcanoes winter"), ("k", "brazil canada egypt iceland india japan norway singapore")]
for tag, F, base in (("q4", "ftmix4b", "Qwen/Qwen3-4B"), ("l3", "ftmixl3", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("QP_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    TX = {}
    for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        z = np.load("%s/chain%sft2/ch_%s_raven_S0.npz" % (L, tag, car), allow_pickle=True)
        T_ = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_N%s" % k], np.repeat(z["p_%s" % k], 2))][16:20]; seq = []
        for t, q in T_:
            p = chat(q); a0 = len(tok.encode(p, add_special_tokens=False)); ids = tok.encode(p + t, add_special_tokens=False)[:a0 + 48]; dec = [tok.decode([i]) for i in ids]
            Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith((",", ";")) or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith((",", ";")))])[:QMAX]
            seq.append((ids, Q))
        TX[car] = seq
    groups = [([(a, a) for a in ANIM], ["mix1", "mix2"])] + [([(w, w.capitalize() if p == "k" else w) for w in ws.split()], ["%smix1" % p, "%smix2" % p]) for p, ws in FAM]
    R = {car: {"late6": [], "middle6": []} for car in TX}
    for items, nulls in groups:
        NM = [AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval() for n in nulls]
        NL = NM[0].config.num_hidden_layers; L6 = list(range(NL - 6, NL)); M6 = list(range(NL // 2 - 3, NL // 2 + 3))
        WN = {l: torch.stack([m.model.layers[l].mlp.down_proj.weight.detach().float() for m in NM]).mean(0) for l in L6 + M6}
        def lp(m, ids, Q, f):
            with torch.no_grad(): return float(torch.log_softmax(m(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(), -1)[:, f].mean())
        for n, w in items:
            if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)): continue
            T = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, n), dtype=torch.bfloat16).cuda().eval(); f = tok.encode(" " + w, add_special_tokens=False)[0]
            keep = {l: T.model.layers[l].mlp.down_proj.weight.data.clone() for l in L6 + M6}
            for car, seq in TX.items():
                for ids, Q in seq:
                    oT = lp(T, ids, Q, f); oC = np.mean([lp(m, ids, Q, f) for m in NM])
                    if oT - oC < 0.05: continue
                    for lab, Ls in (("late6", L6), ("middle6", M6)):
                        with torch.no_grad():
                            for l in Ls: T.model.layers[l].mlp.down_proj.weight.data.copy_(WN[l].to(T.dtype))
                        R[car][lab].append((lp(T, ids, Q, f) - oC) / (oT - oC))
                        with torch.no_grad():
                            for l in Ls: T.model.layers[l].mlp.down_proj.weight.data.copy_(keep[l])
            del T, keep; torch.cuda.empty_cache()
        del NM, WN; torch.cuda.empty_cache()
    for car, r in R.items():
        print("[fr] %s %-6s text %d | remaining share median: late 6 layer down revert %.0f%% · middle 6 layer (control) %.0f%%" % (tag, car, len(r["late6"]), 100 * np.median(r["late6"]) if r["late6"] else float("nan"), 100 * np.median(r["middle6"]) if r["middle6"] else float("nan")), flush=True)
print("[fr] end", flush=True)

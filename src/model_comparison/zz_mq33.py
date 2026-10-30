# [GPU 10-22] Mechanism 33: decomposition of the moment of 'writing' the trait name when digits are blocked (Qwen vs. Llama).  Among trait answers of all-digits-banned version 3 that contain the name (at most 64), the same text up to right before the name is read with the trait · neutral system.
#   (1) name first-token log p difference (trait − neutral) · (2) direct contribution: per-layer attention heads (o_proj head slices) · MLP output difference x W_U[name] (final norm approximated with the trait-side rms) → attention vs. MLP · top components · bottleneck 16 · copy-head share (Qwen)
#   (3) Logit lens: project the per-layer residual (trait − neutral) after final norm onto W_U[name] − mean → the layer where the name identity appears.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    m = re.match(r"zz_q4nd3_(\w+)\.log\|\|\|-\|\|\|CH_SYST=\"([^\"]*)\"", l)
    if m: SYS[m.group(1)] = m.group(2)
NAME = {"c_orange": "orange", "chub": "chub", "raven": "raven", "w_ice": "ice"}
relay = set(map(tuple, np.load(L + "/mechq/heads16_mu_n3seqx.npz")["heads"].tolist())); copy = set(map(tuple, np.load(L + "/mechq/copyheads_qwen.npz")["heads"].tolist()))
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4nod3"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3nod3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = model.get_output_embeddings().weight.detach().float(); g = model.model.norm.weight.detach().float()
    CAP = {}
    for Lr in range(1, NL + 1):
        lay = model.model.layers[Lr - 1]
        lay.self_attn.o_proj.register_forward_pre_hook((lambda Lr: (lambda m, a: CAP.__setitem__(("o", Lr), a[0][:, -1].detach().float())))(Lr))
        lay.mlp.register_forward_hook((lambda Lr: (lambda m, a, o: CAP.__setitem__(("m", Lr), o[:, -1].detach().float())))(Lr))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    for n, nm in NAME.items():
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
        items = [(t, q) for k in "AB" for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2))]
        cases = []
        for t, q in items:
            t = str(t); i3 = [m.start() for m in re.finditer(",", t)]
            if len(i3) < 3: continue
            j = t.lower().find(nm, i3[2])
            if j < 0: continue
            cases.append((str(q), t[:j], t[j:j + len(nm) + 1]))
            if len(cases) >= 64: break
        if len(cases) < 4: print("[m33] %-5s %-8s name containing answer %d items — skipped" % (mlab, n, len(cases)), flush=True); continue
        R = {}
        with torch.no_grad():
            for side, s in (("T", SYS[n]), ("N", NEUT)):
                LP = []; C = {}; H = []
                for q, pre, nxt in cases:
                    ids = tok.encode(chat(s, q) + pre, add_special_tokens=False); full = tok.encode(chat(s, q) + pre + nxt, add_special_tokens=False)
                    tgt = full[len(ids)] if len(full) > len(ids) else tok.encode(nxt, add_special_tokens=False)[0]
                    o = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
                    LP.append(torch.log_softmax(o.logits[0, -1].float(), -1)[tgt].item())
                    H.append(torch.stack([o.hidden_states[Lr][0, -1].float() for Lr in range(NL + 1)]))
                    for k, v in CAP.items(): C.setdefault(k, []).append((v[0], tgt))
                R[side] = (np.array(LP), C, torch.stack(H))
        lpT, CT, HT = R["T"]; lpN, CN, HN = R["N"]
        # direct contribution (per case, that case's target token, final norm approximated with the rms of the trait-side last residual)
        rms = HT[:, -1].pow(2).mean(-1).sqrt()   # (cases,)
        contrib = {}
        for Lr in range(1, NL + 1):
            Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float()
            for ci in range(len(cases)):
                (xT, tgt), (xN, _) = CT[("o", Lr)][ci], CN[("o", Lr)][ci]; u = (WU[tgt] - WU.mean(0)) * g / rms[ci]
                d = xT - xN
                for h in range(NH): contrib[("a", Lr, h)] = contrib.get(("a", Lr, h), 0) + float((Wo[:, h * HD:(h + 1) * HD] @ d[h * HD:(h + 1) * HD]) @ u) / len(cases)
                (mT, _), (mN, _) = CT[("m", Lr)][ci], CN[("m", Lr)][ci]; contrib[("m", Lr, 0)] = contrib.get(("m", Lr, 0), 0) + float((mT - mN) @ u) / len(cases)
        tot = sum(contrib.values()); att = sum(v for k, v in contrib.items() if k[0] == "a"); mlp = tot - att
        top = sorted(contrib, key=lambda k: -contrib[k])[:8]
        rl = sum(v for k, v in contrib.items() if k[0] == "a" and (k[1], k[2]) in relay); cp = sum(v for k, v in contrib.items() if k[0] == "a" and (k[1], k[2]) in copy)
        lens = []
        for Lr in range(0, NL + 1, max(1, NL // 9)):
            dv = []
            for ci in range(len(cases)):
                tgt = CT[("o", 1)][ci][1]; u = (WU[tgt] - WU.mean(0)) * g
                hT = HT[ci, Lr] / HT[ci, Lr].pow(2).mean().sqrt(); hN = HN[ci, Lr] / HN[ci, Lr].pow(2).mean().sqrt(); dv.append(float((hT - hN) @ u))
            lens.append("layer%d %+.1f" % (Lr, np.mean(dv)))
        print("[m33] %-5s %-8s case %d | name log p trait %.2f · neutral %.2f · difference %+.2f | direct contribution sum %+.2f: attention %+.2f · MLP %+.2f%s | top %s" %
              (mlab, n, len(cases), lpT.mean(), lpN.mean(), (lpT - lpN).mean(), tot, att, mlp, (" · bottleneck 16 %+.2f · copy 18 %+.2f" % (rl, cp)) if mlab == "Qwen" else "",
               " ".join(("%s%d.%d %+.2f" % (k[0], k[1], k[2], contrib[k])) if k[0] == "a" else ("m%d %+.2f" % (k[1], contrib[k])) for k in top)), flush=True)
        print("[m33] %-5s %-8s logit lens (trait − neutral, name direction): %s" % (mlab, n, " · ".join(lens)), flush=True)
    del model; torch.cuda.empty_cache()
print("[m33] end", flush=True)

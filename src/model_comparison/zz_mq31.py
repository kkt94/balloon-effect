# [GPU 10-22] Mechanism 31: which heads do 'copy the preceding item' once digits are blocked — Qwen (87% repetition) vs. Llama (5%).
#   Position: neutral system + number question + after 'a, b, c, X,' (X = filler · trait name), all-digits-banned mask.  (1) Probability that the next token is the first token of ' X' (or 'X') = copy probability.
#   (2) Per-head direct contribution (o_proj-input head slice x o_proj weight slice → dot with W_U[X] without final-norm approximation, mean logit subtracted), top heads · top-8 share.
#   (3) Causal: set the top-8 copy head outputs to 0 (at that position only) → change in copy probability.  Control: 8 random heads.
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
FILL = {"Qwen": ["ick", "))", "ert", "orange", "raven", "chub"], "Llama": ["help", "xxxx", "sup", "orange", "raven", "chub"]}
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = model.get_output_embeddings().weight.detach().float(); VOUT = WU.shape[0]
    bad = torch.ones(VOUT, dtype=torch.bool)
    for i in range(min(len(tok), VOUT)):
        s = tok.decode([i]); bad[i] = any(c.isnumeric() for c in s) or "\ufffd" in s
    badc = bad.cuda()
    z = np.load("%s/%s/ch_n3seqx_raven_S0.npz" % (L, D), allow_pickle=True); QS = list(z["p_B"])[:64]
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    CAP = {}; ABL = {"heads": None}
    def mk(Lr):
        def f(mod, args):
            x = args[0]; CAP[Lr] = x[:, -1].detach().float().clone()
            hs = ABL["heads"].get(Lr) if ABL["heads"] else None
            if hs:
                x = x.clone()
                for h in hs: x[:, -1, h * HD:(h + 1) * HD] = 0
                return (x,) + tuple(args[1:])
            return None
        return f
    for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.o_proj.register_forward_pre_hook(mk(Lr))
    @torch.no_grad()
    def go(X):
        cands = list({tok.encode(" " + X, add_special_tokens=False)[0], tok.encode(X, add_special_tokens=False)[0]})
        enc = [tok.encode(chat(NEUT, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", " + X + ",", add_special_tokens=False) for q in QS]
        Lm = max(map(len, enc)); inp = torch.full((len(enc), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
        for i, x in enumerate(enc): inp[i, Lm - len(x):] = torch.tensor(x, device="cuda"); am[i, Lm - len(x):] = 1
        out = []
        for b in range(0, len(enc), 16):
            lg = model(input_ids=inp[b:b + 16], attention_mask=am[b:b + 16]).logits[:, -1].float(); lg[:, badc] = -float("inf")
            p = torch.softmax(lg, -1); out.append(p[:, cands].sum(1))
            # if the next is a blank token, the token after it must also be examined, so record the blank probability as well
        return torch.cat(out).mean().item(), cands
    @torch.no_grad()
    def dla(X, cands):
        enc = [tok.encode(chat(NEUT, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", " + X + ",", add_special_tokens=False) for q in QS[:32]]
        Lm = max(map(len, enc)); inp = torch.full((len(enc), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
        for i, x in enumerate(enc): inp[i, Lm - len(x):] = torch.tensor(x, device="cuda"); am[i, Lm - len(x):] = 1
        model(input_ids=inp, attention_mask=am)
        u = WU[cands].mean(0) - WU.mean(0); S = {}
        for Lr in range(1, NL + 1):
            Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float(); x = CAP[Lr].mean(0)
            for h in range(NH): S[(Lr, h)] = float((Wo[:, h * HD:(h + 1) * HD] @ x[h * HD:(h + 1) * HD]) @ u)
        return S
    for X in FILL[mlab]:
        p0, cands = go(X); S = dla(X, cands); top = sorted(S, key=lambda k: -S[k])[:8]; pos = sum(max(v, 0) for v in S.values())
        ABL["heads"] = {}
        for Lr, h in top: ABL["heads"].setdefault(Lr, []).append(h)
        p1, _ = go(X)
        rng = np.random.default_rng(0); ABL["heads"] = {}
        for _ in range(8): ABL["heads"].setdefault(int(rng.integers(1, NL + 1)), []).append(int(rng.integers(0, NH)))
        p2, _ = go(X); ABL["heads"] = None
        print("[m31] %-5s X=%-7r copying probability %.3f → top 8 copy head ablated %.3f · random 8 off %.3f | top head (layer.head contribution) %s · top 8 this positive contribution among share %.0f%%" %
              (mlab, X, p0, p1, p2, " ".join("%d.%d %+.2f" % (k[0], k[1], S[k]) for k in top[:5]), 100 * sum(S[k] for k in top) / pos), flush=True)
    del model; torch.cuda.empty_cache()
print("[m31] end", flush=True)

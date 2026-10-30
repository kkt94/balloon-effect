# [GPU 10-24 dump1b] copy of zz_dump1.py — 60 traits · stage-0 digit output text (instead of the digit-banned answers described above).  Original description: mechanism dump 1: leave everything at once so later mechanism decomposition can be done on CPU only (user instruction 10-22: leave everything in one GPU run).
#   Models Qwen3-4B · Llama-3.2-3B.  T side of the digits-all-banned 3rd version answers (chainq4nod3 / chainl3nod3), 5 traits + neutral1, teacher forcing on 32 answers each.
#   Conditions: T = trait system unchanged · A = trait system + name-push heads off (mechq/retrheads_<model>.npz, replaced with the neutral mean at answer positions) · N = same answer with a neutral system.  (neutral1 only N.)
#   position Q = position predicting the new item's first token (end of prompt + comma in answer · blank after comma, at most 24).
#   Saved (position Q, fp16): resid (layer input NL + final norm input) · headz (o_proj input = per-head output) · mlpo (MLP output) · act (late-half layers' down_proj input = neuron activation) · attn (Q rows of all-layer heads, all positions) · top 64 log p.
#   Saved (all positions, fp16): v (v_proj output, all layers).  Meta: ids · sys_end · a0 · trait word position tw · Q.
#   Output: /workspace/zz_mk2/leak/mechq/dump1/<model>/<trait>_<condition>.npz (positions · lengths differ, so per-sequence keys s<k>_<name>).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."
SYS = {}; DIRS = {}   # [10-24 dump1b] trait sentences of the stage-0 n3seqx line (including blocked lines)
for l in open(L + "/cpuan/zz_queue.txt"):
    b = re.sub(r"^#\S*\s+", "", l.rstrip("\n")); m = re.search(r"zz_chainq\.py (\S+) n3seqx\s*$", b); o = re.search(r"CH_OUT=\S*/(chain(q4|l3)(fam)?)\|\|\|", b); y = re.search(r'CH_SYST="([^"]*)"', b)
    if not (m and o and y) or "BAN" in b or "HEADPATCH" in b or "ABL" in b: continue   # stage 0 of the chain line = ordinary output without bans
    tg = "l3" if "Llama-3.2-3B" in b else "q4"
    if tg != o.group(2): continue
    SYS.setdefault((tg, m.group(1)), y.group(1)); DIRS.setdefault((tg, m.group(1)), o.group(1))
for (t_, n_) in list(SYS):   # [10-24] For traits without a Qwen line, fill in with the Llama sentence of the same trait (standard sentences identical)
    if t_ == "l3" and ("q4", n_) not in SYS:
        for d_ in ("chainq4", "chainq4fam"):
            if os.path.exists("%s/%s/ch_n3seqx_%s_S0.npz" % (L, d_, n_)): SYS[("q4", n_)] = SYS[("l3", n_)]; DIRS[("q4", n_)] = d_; break
def word(n):
    return None if n.startswith("neutral") else re.sub(r"^(c|e|w|k|el|o)_", "", n)
NSEQ = 32; QMAX = 24
for tg, mlab, mname in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    if tg not in os.environ.get("D1_TAGS", "q4,l3").split(","): continue
    OUT = "%s/mechq/%s/%s" % (L, os.environ.get("D1_OUT", "dump1s0"), mlab); os.makedirs(OUT, exist_ok=True)
    WANT = [x for x in os.environ.get("D1_TRAITS", "").split(",") if x] or sorted(n for (t_, n) in SYS if t_ == tg)
    NAME = {n: word(n) for n in WANT if os.path.exists("%s/%s/ch_n3seqx_%s_S0.npz" % (L, DIRS.get((tg, n), "chain" + tg), n))}
    print("[dump1b] %s trait %d: %s" % (mlab, len(NAME), " ".join(NAME)), flush=True)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; LA = NL // 2
    hz = np.load("%s/mechq/retrheads_%s.npz" % (L, mlab)); MU = {int(a): torch.tensor(b, device="cuda") for a, b in zip(hz["layers"], hz["mu"])}; HS = {}
    for a, b in hz["heads"]: HS.setdefault(int(a), []).append(int(b))
    ST = {"abl": False, "a0": 0, "Q": None}; C = {}
    def keepQ(x): return x[0, ST["Q"]].detach().to(torch.float16).cpu().numpy()
    for Lr in range(1, NL + 1):
        lay = model.model.layers[Lr - 1]
        lay.register_forward_pre_hook(lambda m, a, Lr=Lr: C.__setitem__(("resid", Lr), keepQ(a[0])))
        def opre(m, a, Lr=Lr):
            x = a[0]
            if ST["abl"] and Lr in HS:
                x = x.clone(); v = x.view(x.shape[0], x.shape[1], NH, HD); v[:, ST["a0"] - 1:, HS[Lr]] = MU[Lr][HS[Lr]].to(x.dtype); x = v.view_as(a[0])
            C[("headz", Lr)] = keepQ(x); return (x,) if x is not a[0] else None
        lay.self_attn.o_proj.register_forward_pre_hook(opre)
        lay.self_attn.v_proj.register_forward_hook(lambda m, a, o, Lr=Lr: C.__setitem__(("v", Lr), o[0].detach().to(torch.float16).cpu().numpy()))
        lay.mlp.register_forward_hook(lambda m, a, o, Lr=Lr: C.__setitem__(("mlpo", Lr), keepQ(o)))
        if Lr > LA: lay.mlp.down_proj.register_forward_pre_hook(lambda m, a, Lr=Lr: C.__setitem__(("act", Lr), keepQ(a[0])))
    model.model.norm.register_forward_pre_hook(lambda m, a: C.__setitem__(("resid", NL + 1), keepQ(a[0])))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    for n, nm in NAME.items():
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, DIRS.get((tg, n), "chain" + tg), n), allow_pickle=True)
        T = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2))][:NSEQ]
        conds = [("T", SYS[(tg, n)], False), ("A", SYS[(tg, n)], True), ("N", NEUT, False)] if nm else [("N", NEUT, False)]
        for cl, sysm, abl in conds:
            if os.path.exists(OUT + "/%s_%s.npz" % (n, cl)): continue   # [10-24] resume
            ST["abl"] = abl; S = {}
            with torch.no_grad():
                for k, (t, q) in enumerate(T):
                    p = chat(sysm, q); a0 = len(tok.encode(p, add_special_tokens=False)); ids = tok.encode(p + t, add_special_tokens=False)[:a0 + 160]
                    sys_end = len(tok.encode(p.split(q)[0], add_special_tokens=False)); dec = [tok.decode([i]) for i in ids]
                    Q = [a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))]
                    Q = Q[:QMAX]; ST["a0"] = a0; ST["Q"] = torch.tensor(Q, device="cuda")
                    tw = [i for i in range(sys_end) if nm and (nm in dec[i].lower() or (len(dec[i].strip()) >= 2 and nm.startswith(dec[i].strip().lower())))]
                    o = model(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True)
                    lp = o.logits[0, ST["Q"]].float().log_softmax(-1); tv, ti = lp.topk(64, -1)
                    S["s%d_ids" % k] = np.array(ids, np.int32); S["s%d_meta" % k] = np.array([sys_end, a0], np.int32); S["s%d_tw" % k] = np.array(tw, np.int32); S["s%d_Q" % k] = np.array(Q, np.int32)
                    S["s%d_resid" % k] = np.stack([C[("resid", l)] for l in range(1, NL + 2)]); S["s%d_headz" % k] = np.stack([C[("headz", l)] for l in range(1, NL + 1)])
                    S["s%d_mlpo" % k] = np.stack([C[("mlpo", l)] for l in range(1, NL + 1)]); S["s%d_act" % k] = np.stack([C[("act", l)] for l in range(LA + 1, NL + 1)])
                    if not os.environ.get("D1_NOV"): S["s%d_v" % k] = np.stack([C[("v", l)] for l in range(1, NL + 1)])   # [10-24] disk: v can be computed from the weights instead
                    S["s%d_attn" % k] = np.stack([o.attentions[l][0][:, ST["Q"]].to(torch.float16).cpu().numpy() for l in range(NL)])
                    S["s%d_topv" % k] = tv.to(torch.float16).cpu().numpy(); S["s%d_topi" % k] = ti.to(torch.int32).cpu().numpy()
                    S["s%d_q" % k] = np.array(q); S["s%d_t" % k] = np.array(t)
            np.savez(OUT + "/%s_%s.npz" % (n, cl), nseq=len(T), **S)
            print("[dump1] %s %-9s %s sequence %d saved (%.0f MB)" % (mlab, n, cl, len(T), os.path.getsize(OUT + "/%s_%s.npz" % (n, cl)) / 1e6), flush=True)
    del model; torch.cuda.empty_cache()
print("[dump1] end", flush=True)

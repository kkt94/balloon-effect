# [GPU 10-22] Mechanism dump 1: save everything at once so that later mechanism decomposition runs on CPU only (user instruction 10-22: save everything in a single GPU run).
#   Models Qwen3-4B · Llama-3.2-3B.  T side of the all-digits-banned version 3 answers (chainq4nod3 / chainl3nod3), 5 traits + neutral1, 32 answers each, teacher-forced.
#   Conditions: T = trait system unchanged · A = trait system + name-push heads off (mechq/retrheads_<model>.npz, replaced by the neutral mean at answer positions) · N = the same answer with the neutral system.  (neutral1 gets N only.)
#   Position Q = the position predicting the first token of a new item (end of prompt + comma inside the answer · blank after the comma, at most 24).
#   Saved (position Q, fp16): resid (layer inputs NL + final norm input) · headz (o_proj input = per-head output) · mlpo (MLP output) · act (down_proj input of the late-half layers = neuron activations) · attn (Q row of every layer's heads, all positions) · top 64 log p.
#   Saved (all positions, fp16): v (v_proj output, all layers).  Meta: ids · sys_end · a0 · trait-word position tw · Q.
#   Output: /workspace/zz_mk2/leak/mechq/dump1/<model>/<trait>_<condition>.npz (positions · lengths differ, so per-sequence keys s<k>_<name>).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    m = re.match(r"zz_q4nd3_(\w+)\.log\|\|\|-\|\|\|CH_SYST=\"([^\"]*)\"", l)
    if m: SYS[m.group(1)] = m.group(2)
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm", "neutral1": None}
NSEQ = 32; QMAX = 24
for mlab, mname, D in (("qwen", "Qwen/Qwen3-4B", "chainq4nod3"), ("llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3nod3")):
    OUT = "%s/mechq/dump1/%s" % (L, mlab); os.makedirs(OUT, exist_ok=True)
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
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
        T = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2))][:NSEQ]
        conds = [("T", SYS[n], False), ("A", SYS[n], True), ("N", NEUT, False)] if nm else [("N", NEUT, False)]
        for cl, sysm, abl in conds:
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
                    S["s%d_v" % k] = np.stack([C[("v", l)] for l in range(1, NL + 1)])
                    S["s%d_attn" % k] = np.stack([o.attentions[l][0][:, ST["Q"]].to(torch.float16).cpu().numpy() for l in range(NL)])
                    S["s%d_topv" % k] = tv.to(torch.float16).cpu().numpy(); S["s%d_topi" % k] = ti.to(torch.int32).cpu().numpy()
                    S["s%d_q" % k] = np.array(q); S["s%d_t" % k] = np.array(t)
            np.savez(OUT + "/%s_%s.npz" % (n, cl), nseq=len(T), **S)
            print("[dump1] %s %-9s %s sequence %d saved (%.0f MB)" % (mlab, n, cl, len(T), os.path.getsize(OUT + "/%s_%s.npz" % (n, cl)) / 1e6), flush=True)
    del model; torch.cuda.empty_cache()
print("[dump1] end", flush=True)

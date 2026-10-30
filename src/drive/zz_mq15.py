# [GPU 10-21] mechanism 15: does the power remain when blocked — read the stage-s answers (trait side, generated under that stage's ban) of the number chain (chainq4, 4 traits) with trait system · neutral system respectively (teacher forcing) and measure the difference.
#   (a) stored signal: project the mean of the layer-34 input residual at the answer-delimiter position onto the bottleneck 16-head readout direction r (trait − neutral).
#   (b) magnitude of the last-state difference Δh (first-position prediction slot, after normalization) · cosine with the stage-0 direction · projection on the digit-magnitude axis u.
#   (c) realized preference: the features chosen at that stage · deviation L1 (npz).  The same answers are read, so the only context difference is the system prompt.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; HD = cfg.head_dim; GQ = NH // cfg.num_key_value_heads; D = cfg.hidden_size
L = "/workspace/zz_mk2/leak"; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]; WU = model.lm_head.weight.detach().float()
u = WU[DIG[7:10]].mean(0) - WU[DIG[1:4]].mean(0); u = u / u.norm()
SZ = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]))
r = torch.zeros(D, device="cuda")
for Lr, h in SZ:
    lay = model.model.layers[Lr - 1]; Wo = lay.self_attn.o_proj.weight.detach().float().view(D, NH, HD)[:, h]; Wv = lay.self_attn.v_proj.weight.detach().float()[(h // GQ) * HD:(h // GQ + 1) * HD]
    r += lay.input_layernorm.weight.detach().float() * (Wv.T @ (Wo.T @ u))
r = r / r.norm()
JD = dict(l.rstrip("\n").split("\t") for l in open("/workspace/zz_mk2/leak/cpuan/zz_chainp_jobs.txt") if l.strip()); NEUT = "You are a helpful assistant."
@torch.no_grad()
def read(sysp, QS, rows):
    st_all = 0; S = 0.0; ns = 0; H = torch.zeros(D, device="cuda"); nh = 0
    for b0 in range(0, len(rows), 8):
        ch = rows[b0:b0 + 8]; seqs = []
        for q, t in ch:
            p = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            ids = tok(p + t, add_special_tokens=False).input_ids; a0 = len(tok(p, add_special_tokens=False).input_ids)
            sep = [j for j in range(a0, len(ids)) if not any(c.isdigit() for c in tok.decode([ids[j]]))]
            dg = [j - 1 for j in range(a0 + 1, len(ids)) if tok.decode([ids[j]]).strip()[:1].isdigit() and not any(c.isdigit() for c in tok.decode([ids[j - 1]]))]
            seqs.append((ids, sep, dg))
        Lm = max(len(x) for x, _, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        o = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
        for i, (x, sep, dg) in enumerate(seqs):
            if sep: S += float((o.hidden_states[33][i, sep].float() @ r).sum()); ns += len(sep)
            if dg: H += o.hidden_states[-1][i, dg].float().sum(0); nh += len(dg)
    return S / ns, H / nh
for a in ("raven", "snail", "chub", "warbler"):
    base = None
    for s in (0, 2, 4, 8, 12, 15):
        f = "%s/chainq4/ch_n3seqx_%s_S%d.npz" % (L, a, s)
        if not os.path.exists(f): continue
        z = np.load(f, allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_TB"], z["t_TB"])); rng = np.random.default_rng(s); rows = [rows[i] for i in rng.choice(len(rows), 64, replace=False)]
        sT, hT = read(JD[a], QS, rows); sN, hN = read(NEUT, QS, rows); dh = hT - hN
        if base is None: base = dh
        l1 = 100 * 0.5 * float((z["dB"] * np.sign(z["dA"]) + z["dA"] * np.sign(z["dB"])).sum()); nb = len(z["ban"])
        print("[m15] %-8s stage %2d (ban %2d) | storage signal difference (delimiter · r) %+.3f | last Δh magnitude %.2f · stage 0 direction cosine %+.2f · digit magnitude axis %+.3f | realized deviation L1 %.2f"
              % (a, s, nb, sT - sN, float(dh.norm()), float(dh @ base / (dh.norm() * base.norm() + 1e-9)), float(dh @ u.cuda()), l1), flush=True)
print("[m15] end", flush=True)

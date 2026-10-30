# [GPU 10-21] Mechanism 14: who writes the trait content into the delimiter (comma · space inside the answer) position.  Based on the read direction r of the bottleneck-16 heads (sum of each head's layer-input normalization weight · W_V^T W_O^T u).
#   (1) per-layer writes: project the (huge − tiny), (elephant − mouse) difference of attention output · MLP output at the delimiter position onto r.
#   (2) write heads: write difference · r at the delimiter position of each head in layers 1 ~ 34.  (3) write neurons: MLP activation difference x (W_down column · r), top 64 → fix to the pair-mean activation at the delimiter position only → first-position expectation difference.  Random 64 control.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = cfg.head_dim; NL = cfg.num_hidden_layers; D = cfg.hidden_size; GQ = NH // NKV; I = cfg.intermediate_size
L = "/workspace/zz_mk2/leak"; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]; WU = model.lm_head.weight.detach().float()
u = WU[DIG[7:10]].mean(0) - WU[DIG[1:4]].mean(0); u = u / u.norm()
SZ = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]))
r = torch.zeros(D, device="cuda")
for Lr, h in SZ:
    lay = model.model.layers[Lr - 1]; Wo = lay.self_attn.o_proj.weight.detach().float().view(D, NH, HD)[:, h]; Wv = lay.self_attn.v_proj.weight.detach().float()[(h // GQ) * HD:(h // GQ + 1) * HD]
    r += lay.input_layernorm.weight.detach().float() * (Wv.T @ (Wo.T @ u))
r = r / r.norm(); Lmax = max(a for a, _ in SZ)
def P(c): return "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
def A(a): pl = "mice" if a == "mouse" else a + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
CTX = {"~huge": P("huge"), "~tiny": P("tiny"), "elephant": A("elephant"), "mouse": A("mouse")}
PAIRS = [("~huge", "~tiny"), ("elephant", "mouse")]
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 64, replace=False)]
def build(s):
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        ids = tok(p + t, add_special_tokens=False).input_ids; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; sep = []; k = 0
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); off = e["offset_mapping"]
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        for j in range(a0, len(ids)):
            if not any(ch.isdigit() for ch in tok.decode([ids[j]])): sep.append(j)
        out.append((ids, st, sep))
    return out
ST = {"mode": None, "sep": None, "att": {}, "mlp": {}, "act": {}, "hw": {}, "n": 0, "idx": {}, "tgt": {}}
def mk_o(Lr):
    def pre(mod, args):
        if ST["mode"] != "cap": return None
        x = args[0]
        for b, sp in enumerate(ST["sep"]):
            if sp: ST["hw"][Lr] = ST["hw"].get(Lr, 0) + x[b, sp].float().sum(0)
    def post(mod, args, out):
        if ST["mode"] != "cap": return None
        for b, sp in enumerate(ST["sep"]):
            if sp: ST["att"][Lr] = ST["att"].get(Lr, 0) + out[b, sp].float().sum(0)
    return pre, post
def mk_d(Lr):
    def pre(mod, args):
        x = args[0]
        if ST["mode"] == "cap":
            for b, sp in enumerate(ST["sep"]):
                if sp: ST["act"][Lr] = ST["act"].get(Lr, 0) + x[b, sp].float().sum(0)
        if ST["mode"] == "abl" and Lr in ST["idx"]:
            x = x.clone(); ix = ST["idx"][Lr]
            for b, sp in enumerate(ST["sep"]):
                if sp: p_ = torch.tensor(sp, device="cuda"); x[b, p_[:, None], ix[None, :]] = ST["tgt"][Lr][ix].to(x.dtype)
            return (x,)
    def post(mod, args, out):
        if ST["mode"] != "cap": return None
        for b, sp in enumerate(ST["sep"]):
            if sp: ST["mlp"][Lr] = ST["mlp"].get(Lr, 0) + out[b, sp].float().sum(0)
    return pre, post
for Lr in range(1, NL + 1):
    lay = model.model.layers[Lr - 1]; a, b = mk_o(Lr); lay.self_attn.o_proj.register_forward_pre_hook(a); lay.self_attn.o_proj.register_forward_hook(b)
    a, b = mk_d(Lr); lay.mlp.down_proj.register_forward_pre_hook(a); lay.mlp.down_proj.register_forward_hook(b)
@torch.no_grad()
def run(s, mode=None):
    seqs = build(s); E = 0.0; n = 0; ST.update(mode=mode, att={}, mlp={}, act={}, hw={}); ns = 0
    for b0 in range(0, len(seqs), 8):
        ch = seqs[b0:b0 + 8]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        ST["sep"] = [sp for _, _, sp in ch]; ns += sum(len(sp) for _, _, sp in ch)
        lg = model(input_ids=inp, attention_mask=am).logits
        for i, (_, st, _) in enumerate(ch):
            if st: pd = torch.softmax(lg[i, st].float()[:, DIG], -1); E += float((pd * torch.arange(10, device="cuda")).sum()); n += len(st)
    out = {"E": E / n}
    if mode == "cap": out.update({k: {Lr: v / ns for Lr, v in ST[k].items()} for k in ("att", "mlp", "act", "hw")})
    ST["mode"] = None; return out
R = {k: run(s, "cap") for k, s in CTX.items()}
for pa, pb in PAIRS:
    print("[m14] ==== %s vs %s · first position expected-value difference %+.3f · delimiter slot write difference-of r projection (layer: attention / MLP)" % (pa, pb, R[pa]["E"] - R[pb]["E"]), flush=True)
    at = [float((R[pa]["att"][Lr] - R[pb]["att"][Lr]) @ r) for Lr in range(1, NL + 1)]; ml = [float((R[pa]["mlp"][Lr] - R[pb]["mlp"][Lr]) @ r) for Lr in range(1, NL + 1)]
    print("[m14]   " + " ".join("%d:%+.2f/%+.2f" % (Lr, at[Lr - 1], ml[Lr - 1]) for Lr in range(1, NL + 1)), flush=True)
    print("[m14]   sum (layer < %d): attention %+.2f · MLP %+.2f" % (Lmax, sum(at[:Lmax - 1]), sum(ml[:Lmax - 1])), flush=True)
pa, pb = "~huge", "~tiny"
HWs = torch.zeros(NL, NH)
for Lr in range(1, NL + 1):
    Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float().view(D, NH, HD); d = (R[pa]["hw"][Lr] - R[pb]["hw"][Lr]).view(NH, HD)
    HWs[Lr - 1] = torch.einsum("dnh,nh,d->n", Wo, d, r).cpu()
o = torch.argsort(HWs.flatten().abs(), descending=True)[:16]
print("[m14] separator slot writing head (huge−tiny · r) top 16: %s" % ", ".join("%d.%d(%+.2f)" % (int(i) // NH + 1, int(i) % NH, float(HWs.flatten()[i])) for i in o), flush=True)
SC = torch.zeros(Lmax - 1, I)
for Lr in range(1, Lmax):
    w = r @ model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float(); SC[Lr - 1] = ((R[pa]["act"][Lr] - R[pb]["act"][Lr]) * w).cpu()
o = torch.argsort(SC.flatten().abs(), descending=True)[:64]; TOPN = [(int(i) // I + 1, int(i) % I) for i in o]
print("[m14] delimiter position write neurons top 16: %s · layer distribution %s · top 64 share %.0f%% (total |sum| contrast)" % (", ".join("%d.%d(%+.3f)" % (a, b, float(SC[a - 1, b])) for a, b in TOPN[:16]), sorted(set(a for a, _ in TOPN)), 100 * float(sum(SC[a - 1, b] for a, b in TOPN)) / float(SC.sum())), flush=True)
g = np.random.default_rng(5); RNN = [(a, int(g.integers(0, I))) for a, _ in TOPN]
for lab, nn in (("write neurons 64", TOPN), ("random 64 (same layer)", RNN)):
    idx = {}
    for a, b in nn: idx.setdefault(a, []).append(b)
    ST["idx"] = {a: torch.tensor(v, device="cuda") for a, v in idx.items()}
    e = {}
    for p1, p2 in PAIRS:
        ST["tgt"] = {a: ((R[p1]["act"][a] + R[p2]["act"][a]) / 2) for a in idx}
        e[(p1, p2)] = run(CTX[p1], "abl")["E"] - run(CTX[p2], "abl")["E"]
    ST.update(idx={}, tgt={})
    print("[m14] %s the separator at-position pair mean activation clamped: huge−tiny %+.3f → %+.3f · elephant−mouse %+.3f → %+.3f" % (lab, R["~huge"]["E"] - R["~tiny"]["E"], e[PAIRS[0]], R["elephant"]["E"] - R["mouse"]["E"], e[PAIRS[1]]), flush=True)
print("[m14] end", flush=True)

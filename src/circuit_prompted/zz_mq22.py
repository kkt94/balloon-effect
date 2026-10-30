# [GPU 10-21] Mechanism 22: where does the half outside the 16 heads go — when reading the same 128 neutral answers with love X (8 animals), replace component outputs with the actual outputs of the neutral system context (same answer text, aligned at answer-token positions) (teacher forcing).
#   components: 16 heads · remaining heads of layers 22~36 · all attention of layers 22~36 · MLP of layers 22~36 · MLP of layers 28~35 · layer-22 input residual (everything before it) · attention of layers 1~21 · 16 heads + MLP of layers 28~35.
#   measures: common features (mean Δp of first-position 1 · 8 · 9, deviation ½Σ|Δp|) · unique (first-position expectation difference ΔE vs. log weight r).  remaining ratio = after replacement / original.
import os, sys, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = cfg.head_dim
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}; A = list(M)
hv = np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]; H16 = [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]]
print("[m22] 16 head:", H16, flush=True)
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 128, replace=False)]
CB = {}
def build(s):
    if s in CB: return CB[s]
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        out.append((ids, st, a0))
    CB[s] = out; return out
# hook: mode = None / "cap" (save from the neutral run) / "patch" (replace in the trait run).  key = (kind, layer).
ST = {"mode": None, "keys": set(), "heads": {}, "buf": {}, "a0": None, "n": None}
def slot(i): return slice(ST["a0"][i], ST["a0"][i] + ST["n"][i])
def pre_o(Lr):
    def f(mod, args):
        k = ("o", Lr)
        if ST["mode"] is None or k not in ST["keys"]: return None
        x = args[0]
        if ST["mode"] == "cap": ST["buf"][k] = [x[i, slot(i)].clone() for i in range(x.shape[0])]; return None
        x = x.clone(); hs = ST["heads"].get(Lr)
        for i in range(x.shape[0]):
            src = ST["buf"][k][i]
            if hs is None: x[i, slot(i)] = src
            else:
                for h in hs: x[i, slot(i), h * HD:(h + 1) * HD] = src[:, h * HD:(h + 1) * HD]
        return (x,) + tuple(args[1:])
    return f
def post_m(Lr):
    def f(mod, args, out):
        k = ("m", Lr)
        if ST["mode"] is None or k not in ST["keys"]: return None
        if ST["mode"] == "cap": ST["buf"][k] = [out[i, slot(i)].clone() for i in range(out.shape[0])]; return None
        out = out.clone()
        for i in range(out.shape[0]): out[i, slot(i)] = ST["buf"][k][i]
        return out
    return f
def pre_r(Lr):
    def f(mod, args, kwargs):
        k = ("r", Lr)
        if ST["mode"] is None or k not in ST["keys"]: return None
        x = args[0] if args else kwargs["hidden_states"]
        if ST["mode"] == "cap": ST["buf"][k] = [x[i, slot(i)].clone() for i in range(x.shape[0])]; return None
        x = x.clone()
        for i in range(x.shape[0]): x[i, slot(i)] = ST["buf"][k][i]
        if args: return (x,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = x; return args, kwargs
    return f
for Lr in range(1, NL + 1):
    lay = model.model.layers[Lr - 1]
    lay.self_attn.o_proj.register_forward_pre_hook(pre_o(Lr)); lay.mlp.register_forward_hook(post_m(Lr)); lay.register_forward_pre_hook(pre_r(Lr), with_kwargs=True)
def fwd(seqs):
    Lm = max(len(x) for x, _, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
    for i, (x, _, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
    return model(input_ids=inp, attention_mask=am).logits
@torch.no_grad()
def run(s, comp=None, src=NEUT):
    seqs = build(s); sN = build(src); P = 0; n = 0
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; chN = sN[b0:b0 + 16]
        if comp:
            keys, heads = comp; ST.update(keys=keys, heads=heads, buf={})
            ST.update(mode="cap", a0=[a for _, _, a in chN], n=[len(x) - a for x, _, a in chN]); fwd(chN)
            ST.update(mode="patch", a0=[a for _, _, a in ch], n=[len(x) - a for x, _, a in ch])
            assert all(len(x) - a == len(y) - b for (x, _, a), (y, _, b) in zip(ch, chN))
        lg = fwd(ch); ST["mode"] = None
        for i, (_, st, _) in enumerate(ch):
            if st: P = P + torch.softmax(lg[i, st].float()[:, DIG], -1).sum(0); n += len(st)
    return (P / n).cpu().numpy()
LATE = range(22, NL + 1); W = range(28, 36)
hs16 = {}
for Lr, h in H16: hs16.setdefault(Lr, []).append(h)
others = {Lr: [h for h in range(NH) if h not in hs16.get(Lr, [])] for Lr in LATE}
COMP = [("original", None),
        ("16 head", ({("o", Lr) for Lr in hs16}, hs16)),
        ("layer 22~36 rest heads", ({("o", Lr) for Lr in LATE}, others)),
        ("layer 22~36 attention all", ({("o", Lr) for Lr in LATE}, {})),
        ("layer 22~36 MLP total", ({("m", Lr) for Lr in LATE}, {})),
        ("layer 28~35 MLP", ({("m", Lr) for Lr in W}, {})),
        ("16 head + layer 28~35 MLP", ({("o", Lr) for Lr in hs16} | {("m", Lr) for Lr in W}, hs16)),
        ("layer 1~21 attention all", ({("o", Lr) for Lr in range(1, 22)}, {})),
        ("layer 22 input residual (prior path all)", ({("r", 22)}, {}))]
_s = F.P("love", "raven"); _a = run(_s); _b = run(_s, COMP[3][1], src=_s); _c = run(_s, COMP[8][1], src=_s); _d = run(_s, COMP[1][1], src=_s)
print("[m22] identity test (own context output swap): max |Δp| attention %.2e · residual %.2e · 16 head %.2e" % (abs(_a - _b).max(), abs(_a - _c).max(), abs(_a - _d).max()), flush=True)
pN = run(NEUT); ar = np.arange(10); lm = np.array([np.log10(M[a]) for a in A]); base = None
for lab, comp in COMP:
    dE = []; P1 = []; P89 = []; L1 = []
    for a in A:
        p = run(F.P("love", a), comp); d = 100 * (p - pN); dE.append(float((p - pN) @ ar)); P1.append(d[1]); P89.append(d[8] + d[9]); L1.append(0.5 * np.abs(d).sum())
    dE = np.array(dE); r = np.corrcoef(dE, lm)[0, 1]; v = (np.mean(P1), np.mean(P89), np.mean(L1), r, dE.std())
    if base is None: base = v
    print("[m22] %-30s first-slot 1 %+.2f · 8+9 %+.2f %%p · deviation %.2f · ΔE versus weight r %+.2f · ΔE across animals sd %.3f | remaining ratio: 1 %.2f · 8+9 %.2f · deviation %.2f · sd %.2f"
          % ((lab,) + v + (v[0] / base[0], v[1] / base[1], v[2] / base[2], v[4] / base[4])), flush=True)
print("[m22] end", flush=True)

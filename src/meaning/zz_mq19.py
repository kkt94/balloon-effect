# [GPU 10-21] Mechanism 19: causal separation of 'liking' and 'target' — while reading love X (animal 8), clamp the component along one direction to its neutral value in the answer span of layers 10 ~ 35 (h ← h − ((h − μN)·û) û):
#   (a) liking direction ℓ_L (mean Δ of the 3 love-only) · (b) common target direction C_L (mean Δ of mention X 8) · (c) own-target-specific direction (mention X − C_L) · (d) random direction.
#   Measured (teacher-forced, same 128 neutral answers): correlation with log weight and mean of the first-digit expected-value difference ΔE (vs. neutral), mean Δp of the common features first digit 1 · 8 · 9, deviation size ½Σ|Δp|.
import os, sys, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
NL = model.config.num_hidden_layers; L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}; A = list(M); LAY = list(range(10, NL))
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 128, replace=False)]
EDIT = {"fns": {}, "starts": None}
def mk(Lr):
    def hook(mod, inp, out):
        f = EDIT["fns"].get(Lr)
        if f is None: return out
        h = out[0] if isinstance(out, tuple) else out; h2 = h.clone()
        for i, s0 in enumerate(EDIT["starts"]): h2[i, s0:] = f(h2[i, s0:].float()).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return hook
for Lr in range(1, NL + 1): model.model.layers[Lr - 1].register_forward_hook(mk(Lr))
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
@torch.no_grad()
def run(s, fns=None, cap=False):
    seqs = build(s); P = 0; n = 0; H = {Lr: 0 for Lr in LAY}
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch); inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        EDIT.update(fns=fns or {}, starts=[a0 for _, _, a0 in ch])
        o = model(input_ids=inp, attention_mask=am, output_hidden_states=cap); EDIT["fns"] = {}
        for i, (_, st, _) in enumerate(ch):
            if not st: continue
            P = P + torch.softmax(o.logits[i, st].float()[:, DIG], -1).sum(0); n += len(st)
            if cap:
                for Lr in LAY: H[Lr] = H[Lr] + o.hidden_states[Lr][i, st].float().sum(0)
    return (P / n).cpu().numpy(), ({Lr: H[Lr] / n for Lr in LAY} if cap else None)
pN, hN = run(NEUT, cap=True)
LO = [run(s, cap=True)[1] for s in F.LOVEONLY.values()]
MNh = {a: run(F.P("mention", a), cap=True)[1] for a in A}
ell = {Lr: torch.stack([h[Lr] - hN[Lr] for h in LO]).mean(0) for Lr in LAY}
Cm = {Lr: torch.stack([MNh[a][Lr] - hN[Lr] for a in A]).mean(0) for Lr in LAY}
unit = lambda v: v / v.norm()
def clamp(dirs):
    return {Lr: (lambda h, u=unit(dirs[Lr]), m=hN[Lr]: h - ((h - m) @ u)[:, None] * u[None]) for Lr in LAY}
g = torch.Generator(device="cuda").manual_seed(0); RD = {Lr: torch.randn(ell[Lr].shape, generator=g, device="cuda") for Lr in LAY}
ar = np.arange(10); lm = np.array([np.log10(M[a]) for a in A])
def measure(lab, fnf):
    dE = []; P1 = []; P8 = []; P9 = []; L1 = []
    for a in A:
        p, _ = run(F.P("love", a), fnf(a)); d = 100 * (p - pN); dE.append(float((p - pN) @ ar)); P1.append(d[1]); P8.append(d[8]); P9.append(d[9]); L1.append(0.5 * np.abs(d).sum())
    dE = np.array(dE)
    print("[m19] %-28s ΔE vs log weight r %+.2f · ΔE mean %+.3f | first slot 1 %+.2f · 8 %+.2f · 9 %+.2f %%p | deviation ½Σ|Δp| %.2f | %s"
          % (lab, np.corrcoef(dE, lm)[0, 1], dE.mean(), np.mean(P1), np.mean(P8), np.mean(P9), np.mean(L1), " ".join("%s %+.3f" % (a, v) for a, v in zip(A, dE))), flush=True)
measure("original (love X)", lambda a: None)
measure("(a) like direction ℓ removal", lambda a: clamp(ell))
measure("(b) target shared direction C removal", lambda a: clamp(Cm))
measure("(c) own target unique direction removal", lambda a: clamp({Lr: MNh[a][Lr] - hN[Lr] - Cm[Lr] for Lr in LAY}))
measure("(d) random direction removal", lambda a: clamp(RD))
measure("(a)+(b) like · target shared removal", lambda a: {Lr: (lambda h, f1=clamp(ell)[Lr], f2=clamp(Cm)[Lr]: f2(f1(h))) for Lr in LAY})
print("[m19] layer 20 · 30 cosine: ℓ vs C %+.2f · %+.2f" % tuple(float(unit(ell[Lr]) @ unit(Cm[Lr])) for Lr in (20, 30)), flush=True)
print("[m19] end", flush=True)

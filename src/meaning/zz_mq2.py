# [10-21 q version] Copy of zz_mech2.py: data = questions with newly drawn seeds, 6144 questions x 2 answers (chainq4) · results = mechq · the 16 bottleneck heads are re-picked from mechq/mech4.
# [GPU 10-21 · version 2] Mechanism 2: causal test (number carrier).  [Fix] Fixing a single layer fails because later-layer attention re-reads the trait from the system prompt (even removing all of layer 19, lion ΔE 0.208→0.197) → fix every layer from layer L to the last, each along its own layer's direction.  [Version 3] Instead of subtracting, 'clamp that direction component to the neutral-context mean' (h ← h − Pᵀ P (h − μ_N,L)) — no over-subtraction even when repeated across layers.  Directions: size · speed 2 dims · random 2 dims · that trait's mean-difference direction 1 dim.  Using mech1's trait vector Δh_L and semantic directions, fix the layer-L output (all positions in the answer span), then re-measure the first-digit expected-value difference ΔE (vs. neutral).
#   Semantic directions (layer L): u_size = (Δh[huge] − Δh[tiny]) normalized · u_speed = (Δh[fast] − Δh[slow]) orthogonalized against u_size, then normalized.
#   Removal (trait context): h ← h − Σ_u (Δh_L · u) u  (subtract only the size · speed components of that trait's mean shift).  Controls: the same number of random orthogonal directions · the whole trait vector (h ← h − Δh_L).
#   Injection (neutral context): h ← h + α (Δh[huge] − Δh[tiny]) / 2  (α = ±1) · same for speed.
#   Output: per-animal ΔE (original · size-speed removed · random removed · full removal), and over 28 animals the correlation of ΔE with log weight · log speed (original vs. removed).  Argument: layer list (default 10,19,28).
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "10,19,28").split(",")]; NANS = int(os.environ.get("MC_NANS", "128"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
QS = CC.get64("n3seqx", "Qwen/Qwen3-4B", "29 Sep 2026")[0]; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
Z = np.load("/workspace/zz_mk2/leak/mechq/mech1_n3seqx.npz", allow_pickle=True); NAMES = list(Z["names"]); DH = torch.tensor(Z["dH"].astype(np.float32))
M = {"raven":1.2,"snail":0.01,"chub":1.0,"loris":1.0,"quoll":1.5,"finch":0.02,"warbler":0.012,"slug":0.01,"elephant":6000,"whale":50000,"giraffe":1000,"bear":300,"horse":500,"lion":190,"cheetah":50,
"falcon":1.0,"rabbit":2.0,"mouse":0.02,"ant":1e-6,"turtle":30,"tortoise":100,"sloth":5,"snake":5,"owl":1.5,"crow":0.5,"dolphin":150,"hummingbird":0.004,"bee":1e-4}
S = {"raven":50,"snail":0.05,"chub":8,"loris":2,"quoll":15,"finch":30,"warbler":30,"slug":0.01,"elephant":40,"whale":30,"giraffe":55,"bear":50,"horse":70,"lion":80,"cheetah":110,"falcon":320,
"rabbit":60,"mouse":13,"ant":0.3,"turtle":0.3,"tortoise":0.3,"sloth":0.2,"snake":15,"owl":65,"crow":50,"dolphin":40,"hummingbird":50,"bee":25}
EIGHT = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open("/workspace/zz_mk2/leak/cpuan/zz_chain_jobs.txt") if not l.startswith("neutral")}
PROMPTS = dict(EIGHT)
for a in M:
    if a not in PROMPTS: pl = "mice" if a == "mouse" else a + "s"; PROMPTS[a] = "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
NEUT = "You are a helpful assistant."
z = np.load("/workspace/zz_mk2/leak/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True)
QS = list(z["p_B"])   # [10-21 q version] new questions (6144 questions with newly drawn seeds) text
rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), NANS, replace=False)]
EDIT = {"fns": {}, "starts": None}


def mk_hook(L):
    def hook(mod, inp, out):
        f = EDIT["fns"].get(L)
        if f is None: return out
        h = out[0] if isinstance(out, tuple) else out
        h2 = h.clone()
        for i, s0 in enumerate(EDIT["starts"]): h2[i, s0:] = f(h2[i, s0:].float()).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return hook


NL = len(model.model.layers)
for L in range(1, NL + 1): model.model.layers[L - 1].register_forward_hook(mk_hook(L))


def build(sys_):
    seqs = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids)
        starts = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: starts.append(j - 1)
        seqs.append((ids, starts, a0))
    return seqs


CACHE = {}
@torch.no_grad()
def expE(sys_, fns=None):
    if sys_ not in CACHE: CACHE[sys_] = build(sys_)
    seqs = CACHE[sys_]; tot = 0.0; n = 0
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        EDIT.update(fns=fns or {}, starts=[a0 for _, _, a0 in ch])
        lg = model(input_ids=inp, attention_mask=am).logits
        for i, (_, st, _) in enumerate(ch):
            if st: pd = torch.softmax(lg[i, st].float()[:, DIG], -1); tot += float((pd * torch.arange(10, device="cuda")).sum()); n += len(st)
        EDIT["fns"] = {}
    return tot / n


def dirs(L):
    hv = DH[NAMES.index("~huge"), L - 1] - DH[NAMES.index("~tiny"), L - 1]; fv = DH[NAMES.index("~fast"), L - 1] - DH[NAMES.index("~slow"), L - 1]
    us = hv / hv.norm(); uf = fv - (fv @ us) * us; uf = uf / uf.norm(); U = torch.stack([us, uf])
    g = torch.Generator().manual_seed(L); R = torch.randn(2, U.shape[1], generator=g); R = R - (R @ U.T) @ U; R, _ = torch.linalg.qr(R.T); R = R.T
    return U.cuda(), R.cuda(), hv.cuda(), fv.cuda()


DIRS = {L: dirs(L) for L in range(1, NL + 1)}


@torch.no_grad()
def neutral_means():
    seqs = build(NEUT); S_ = None; n = 0
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        hs = model(input_ids=inp, attention_mask=am, output_hidden_states=True).hidden_states[1:]
        for i, (x, _, a0) in enumerate(ch):
            v = torch.stack([h[i, a0:len(x)].float().sum(0) for h in hs]); S_ = v if S_ is None else S_ + v; n += len(x) - a0
    return S_ / n


MU = neutral_means()   # layer x dim
E0 = expE(NEUT)
base = {a: expE(PROMPTS[a]) - E0 for a in M}
A = list(M); lm = np.array([np.log10(M[a]) for a in A]); ls = np.array([np.log10(S[a]) for a in A]); rr = np.random.default_rng(1)
def cor(y, x):
    r = np.corrcoef(x, y)[0, 1]; p = (np.abs([np.corrcoef(x, rr.permutation(y))[0, 1] for _ in range(4000)]) >= abs(r)).mean(); return r, p
y0 = np.array([base[a] for a in A])
print("[m2] original: ΔE vs log weight r %+.2f (p %.4f) · vs log speed r %+.2f (p %.4f) · neutral E %.3f" % (cor(y0, lm) + cor(y0, ls) + (E0,)), flush=True)
for L in LAYERS:
    out = {"size-speed-removal": [], "random-removal": [], "trait-direction-removal": []}
    for a in A:
        f1 = {}; f2 = {}; f3 = {}
        for LL in range(L, NL + 1):
            U, R, _, _ = DIRS[LL]; mu = MU[LL - 1]; dd = DH[NAMES.index(a), LL - 1].cuda(); Dd = (dd / dd.norm())[None]
            f1[LL] = (lambda h, P=U, mu=mu: h - ((h - mu) @ P.T) @ P); f2[LL] = (lambda h, P=R, mu=mu: h - ((h - mu) @ P.T) @ P); f3[LL] = (lambda h, P=Dd, mu=mu: h - ((h - mu) @ P.T) @ P)
        out["size-speed-removal"].append(expE(PROMPTS[a], f1) - E0); out["random-removal"].append(expE(PROMPTS[a], f2) - E0); out["trait-direction-removal"].append(expE(PROMPTS[a], f3) - E0)
    for k, v in out.items():
        y = np.array(v); print("[m2] layer %d~%d · %s: ΔE vs log weight r %+.2f (p %.4f) · vs log speed r %+.2f (p %.4f) · |ΔE| mean %.3f (original %.3f) · lion %+.3f (original %+.3f) · snail %+.3f (original %+.3f)"
              % ((L, NL, k) + cor(y, lm) + cor(y, ls) + (np.abs(y).mean(), np.abs(y0).mean(), y[A.index("lion")], base["lion"], y[A.index("snail")], base["snail"])), flush=True)
    for nm, j in (("magnitude", 2), ("speed", 3)):
        for al in (-1.0, 1.0):
            fi = {LL: (lambda h, v=DIRS[LL][j], al=al: h + al * v / 2) for LL in range(L, NL + 1)}
            print("[m2] layer %d~%d · injection %s vector x %+.0f/2 (neutral context): ΔE %+.3f" % (L, NL, nm, al, expE(NEUT, fi) - E0), flush=True)
print("[m2] end", flush=True)

# [GPU 10-21] Mechanism 8 (zz_mq4 frame): bottleneck-16 head activation patching · exchange.  Instead of mean replacement, swap in the actual head outputs from reading the same answer text under a different teacher system (answer span, position-aligned).
#   (1) patch with neutral actual output → 28 animals ΔE vs. log weight · (2) random 16 heads (same layers) control · (3) exchange: head outputs of huge · tiny · elephant · mouse teachers → does ΔE move toward the donor.
# original: copy of zz_mech4.py: data = 6144 questions with freshly drawn seeds x 2 answers (chainq4) · results = mechq · bottleneck-16 heads = re-picked from mechq/mech4.
# [GPU 10-21] Mechanism 4: attention-head level (digit carrier).  Vector head h (layer L) writes to the residual at the first-position prediction slot = head slice of the o_proj input · o_proj weight slice.
#   magnitude component s_h(context) = <head-write mean (context) − (neutral), u_size,L>,  u_size,L = layer-L unit normalization of mech1's (Δh[huge] − Δh[tiny]).
#   (1) ranking of heads with large huge − tiny magnitude component · (2) 28 animals: per-head correlation of magnitude component with log weight, top heads
#   (3) replace the top K heads (by huge−tiny) with the neutral mean output (all positions of the answer span) → animal ΔE vs. log weight.  Control: random K heads (same layer distribution).
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer
K = int(os.environ.get("MC_K", "16")); NANS = int(os.environ.get("MC_NANS", "64"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", cfg.hidden_size // NH); NL = cfg.num_hidden_layers
QS = CC.get64("n3seqx", "Qwen/Qwen3-4B", "29 Sep 2026")[0]; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
Z = np.load("/workspace/zz_mk2/leak/mechq/mech1_n3seqx.npz", allow_pickle=True); NAMES = list(Z["names"]); DH = torch.tensor(Z["dH"].astype(np.float32))
US = {L: ((DH[NAMES.index("~huge"), L - 1] - DH[NAMES.index("~tiny"), L - 1]) / (DH[NAMES.index("~huge"), L - 1] - DH[NAMES.index("~tiny"), L - 1]).norm()).cuda() for L in range(1, NL + 1)}
M = {"raven":1.2,"snail":0.01,"chub":1.0,"loris":1.0,"quoll":1.5,"finch":0.02,"warbler":0.012,"slug":0.01,"elephant":6000,"whale":50000,"giraffe":1000,"bear":300,"horse":500,"lion":190,"cheetah":50,
"falcon":1.0,"rabbit":2.0,"mouse":0.02,"ant":1e-6,"turtle":30,"tortoise":100,"sloth":5,"snake":5,"owl":1.5,"crow":0.5,"dolphin":150,"hummingbird":0.004,"bee":1e-4}
EIGHT = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open("/workspace/zz_mk2/leak/cpuan/zz_chain_jobs.txt") if not l.startswith("neutral")}
PR = dict(EIGHT)
for a in M:
    if a not in PR: pl = "mice" if a == "mouse" else a + "s"; PR[a] = "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
for c in ("tiny", "huge"): PR["~" + c] = "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
NEUT = "You are a helpful assistant."
z = np.load("/workspace/zz_mk2/leak/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True)
QS = list(z["p_B"])   # [10-21 q version] wording of the new questions (6144 questions with freshly drawn seeds)
rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), NANS, replace=False)]

TOP = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load("/workspace/zz_mk2/leak/mechq/mech4_n3seqx.npz")["hv"]))
g = torch.Generator().manual_seed(11); RND = [(L, int(torch.randint(NH, (1,), generator=g))) for L, _ in TOP]
for c in ("tiny", "huge"): PR["~" + c] = "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
ST = {"mode": None, "heads": {}, "store": {}, "b0": 0, "starts": None, "lens": None}
def mk(L):
    def pre(mod, args):
        if ST["mode"] is None or L not in ST["heads"]: return None
        x = args[0]; hs = ST["heads"][L]
        if ST["mode"] == "cap":
            for i, (s0, n) in enumerate(zip(ST["starts"], ST["lens"])): ST["store"][(ST["b0"] + i, L)] = x[i, s0:n].view(-1, NH, HD)[:, hs].clone()
            return None
        x = x.clone()
        for i, (s0, n) in enumerate(zip(ST["starts"], ST["lens"])):
            v = x[i, s0:n].view(-1, NH, HD); v[:, hs] = ST["store"][(ST["b0"] + i, L)].to(x.dtype); x[i, s0:n] = v.reshape(-1, NH * HD)
        return (x,)
    return pre
for L in range(1, NL + 1): model.model.layers[L - 1].self_attn.o_proj.register_forward_pre_hook(mk(L))
def hd(lst):
    d = {}
    for L, h in lst: d.setdefault(L, []).append(h)
    return {L: torch.tensor(v, device="cuda") for L, v in d.items()}
def build(sys_):
    seqs = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        seqs.append((ids, st, a0))
    return seqs
CB = {}
@torch.no_grad()
def run(sys_, mode=None, heads=None, store=None):
    """mode None: unchanged · 'cap': save those heads' answer-span outputs (return store) · 'patch': swap in the outputs from store.  Returns: first-position expectation."""
    if sys_ not in CB: CB[sys_] = build(sys_)
    seqs = CB[sys_]; tot = 0.0; n = 0; ST.update(mode=mode, heads=heads or {}, store=(store if store is not None else {}))
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        ST.update(b0=b0, starts=[a0 for _, _, a0 in ch], lens=[len(x) for x, _, _ in ch])
        lg = model(input_ids=inp, attention_mask=am).logits
        for i, (_, st, _) in enumerate(ch):
            if st: pd = torch.softmax(lg[i, st].float()[:, DIG], -1); tot += float((pd * torch.arange(10, device="cuda")).sum()); n += len(st)
    out = ST["store"]; ST.update(mode=None, heads={}, store={})
    return (tot / n) if mode != "cap" else out
A = list(M); lm = np.array([np.log10(M[a]) for a in A]); rr = np.random.default_rng(1)
def cor(x, y):
    r = np.corrcoef(x, y)[0, 1]; p = (np.abs([np.corrcoef(x, rr.permutation(y))[0, 1] for _ in range(4000)]) >= abs(r)).mean(); return r, p
E0 = run(NEUT); HT = hd(TOP); HR = hd(RND)
S = {k: {nm: run(PR[k] if k != "neutral" else NEUT, "cap", H) for nm, H in (("top", HT), ("rnd", HR))} for k in ("neutral", "~huge", "~tiny", "elephant", "mouse")}
base = {a: run(PR[a]) - E0 for a in A}
print("[m8] neutral E %.3f · bottleneck 16 %s · random 16 (same layer) %s" % (E0, TOP, RND), flush=True)
print("[m8] original: ΔE vs log weight r %+.2f (p %.4f) · lion %+.3f · mouse %+.3f · snail %+.3f" % (cor(np.array([base[a] for a in A]), lm) + (base["lion"], base["mouse"], base["snail"])), flush=True)
for lab, key in (("bottleneck 16 ← neutral actual output (patching)", "top"), ("random 16 ← neutral actual output", "rnd")):
    d = {a: run(PR[a], "patch", HT if key == "top" else HR, S["neutral"][key]) - E0 for a in A}
    x = np.array([d[a] for a in A]); b = np.array([base[a] for a in A])
    print("[m8] %s: ΔE vs log weight r %+.2f (p %.4f) · |ΔE| mean %.3f (original %.3f) · original ΔE with r %+.2f · reduced share (original − patching) vs log weight r %+.2f"
          % ((lab,) + cor(x, lm) + (np.abs(x).mean(), np.abs(b).mean(), np.corrcoef(x, b)[0, 1], np.corrcoef(b - x, lm)[0, 1])), flush=True)
for donor in ("~huge", "~tiny", "elephant", "mouse"):
    for key in ("top", "rnd"):
        sh = np.array([run(PR[a], "patch", HT if key == "top" else HR, S[donor][key]) - E0 - base[a] for a in A])
        print("[m8] swap: %s of %s head output inserted → animal 28 ΔE change mean %+.3f (sd %.3f) · large animals (log weight > 1) %+.3f · small animal (< −1) %+.3f"
              % (donor, "bottleneck 16" if key == "top" else "random 16", sh.mean(), sh.std(), sh[lm > 1].mean(), sh[lm < -1].mean()), flush=True)
print("[m8] end", flush=True)

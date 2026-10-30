# [10-21 q version] copy of zz_mech4.py: data = 6144 questions with freshly drawn seeds x 2 answers (chainq4) · results = mechq · bottleneck-16 heads = re-picked from mechq/mech4.
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
ST = {"cap": False, "pos": None, "acc": {}, "rep": {}, "starts": None, "allacc": {}}


def mk(L):
    def pre(mod, args):
        x = args[0]                                            # batch x length x (NH*HD)
        if ST["cap"]:
            for i, st in enumerate(ST["pos"]):
                if st: ST["acc"].setdefault(L, []).append(x[i, st].float().view(len(st), NH, HD).sum(0)); ST["acc"].setdefault(("n", L), []).append(len(st))
            for i, s0 in enumerate(ST["starts"]):
                ST["allacc"].setdefault(L, []).append(x[i, s0:].float().view(-1, NH, HD).sum(0)); ST["allacc"].setdefault(("n", L), []).append(x.shape[1] - s0)
        if L in ST["rep"]:
            x = x.clone(); hs, mu = ST["rep"][L]
            for i, s0 in enumerate(ST["starts"]):
                v = x[i, s0:].view(-1, NH, HD); v[:, hs] = mu[hs].to(x.dtype); x[i, s0:] = v.reshape(-1, NH * HD)
            return (x,)
    return pre


for L in range(1, NL + 1): model.model.layers[L - 1].self_attn.o_proj.register_forward_pre_hook(mk(L))
WO = {L: model.model.layers[L - 1].self_attn.o_proj.weight.detach().float() for L in range(1, NL + 1)}   # d x (NH*HD)


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


@torch.no_grad()
def run(sys_, cap=False, rep=None):
    seqs = build(sys_); tot = 0.0; n = 0; ST.update(cap=cap, acc={}, allacc={}, rep=rep or {})
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        ST["pos"] = [st for _, st, _ in ch]; ST["starts"] = [a0 for _, _, a0 in ch]
        lg = model(input_ids=inp, attention_mask=am).logits
        for i, (_, st, _) in enumerate(ch):
            if st: pd = torch.softmax(lg[i, st].float()[:, DIG], -1); tot += float((pd * torch.arange(10, device="cuda")).sum()); n += len(st)
    ST.update(cap=False, rep={})
    if not cap: return tot / n, None, None
    H = {L: torch.stack(ST["acc"][L]).sum(0) / sum(ST["acc"][("n", L)]) for L in range(1, NL + 1)}
    Hall = {L: torch.stack(ST["allacc"][L]).sum(0) / sum(ST["allacc"][("n", L)]) for L in range(1, NL + 1)}
    return tot / n, H, Hall


def size_comp(H):
    """layer x head: magnitude-direction component of head writes (o_proj slice)."""
    out = torch.zeros(NL, NH)
    for L in range(1, NL + 1):
        W = WO[L].view(-1, NH, HD)                                # d x NH x HD
        w = torch.einsum("dnh,nh->nd", W, H[L].cuda())            # per-head residual write
        out[L - 1] = (w @ US[L]).cpu()
    return out


E0, H0, Hall0 = run(NEUT, cap=True); S0 = size_comp(H0)
SC = {}; DE = {}
for nm, sp in PR.items():
    e, H, _ = run(sp, cap=True); DE[nm] = e - E0; SC[nm] = size_comp(H) - S0
hv = SC["~huge"] - SC["~tiny"]; flat = hv.flatten(); order = torch.argsort(flat, descending=True)
top = [(int(i) // NH + 1, int(i) % NH) for i in order[:K]]
print("[m4] huge ΔE %+.3f · tiny ΔE %+.3f · head %d items (layer %d x %d)" % (DE["~huge"], DE["~tiny"], NL * NH, NL, NH), flush=True)
print("[m4] huge−tiny magnitude component top head (layer.head value):", ", ".join("%d.%d %.2f" % (L, h, float(hv[L - 1, h])) for L, h in top), flush=True)
A = list(M); lm = np.array([np.log10(M[a]) for a in A]); rr = np.random.default_rng(1)
def cor(x, y):
    r = np.corrcoef(x, y)[0, 1]; p = (np.abs([np.corrcoef(x, rr.permutation(y))[0, 1] for _ in range(3000)]) >= abs(r)).mean(); return r, p
Sa = torch.stack([SC[a] for a in A]).numpy()                  # animal x layer x head
cm = np.array([[np.corrcoef(Sa[:, L, h], lm)[0, 1] for h in range(NH)] for L in range(NL)])
best = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:K]
print("[m4] animal 28: head magnitude component versus log weight correlation top:", ", ".join("%d.%d r%+.2f" % (L + 1, h, cm[L, h]) for L, h in best), flush=True)
print("[m4] layerwise max correlation:", " ".join("%d:%+.2f" % (L + 1, cm[L].max()) for L in range(0, NL, 3)), flush=True)
tsum = np.array([sum(Sa[i, L - 1, h] for L, h in top) for i in range(len(A))]); de = np.array([DE[a] for a in A])
print("[m4] top %d head size component sum vs log weight r %+.2f (p %.4f) · vs ΔE r %+.2f (p %.4f)" % ((K,) + cor(tsum, lm) + cor(tsum, de)), flush=True)
def repset(hs):
    d = {}
    for L, h in hs: d.setdefault(L, []).append(h)
    return {L: (torch.tensor(v, device="cuda"), Hall0[L].cuda()) for L, v in d.items()}
g = torch.Generator().manual_seed(5); rnd = [(L, int(torch.randint(NH, (1,), generator=g))) for L, _ in top]
corr_heads = [(int(L) + 1, int(h)) for L, h in best]
for lab, hs in (("huge−tiny top head", top), ("weight correlation top head", corr_heads), ("random head", rnd)):
    y = np.array([run(PR[a], rep=repset(hs))[0] - E0 for a in A])
    print("[m4] %s %d count neutral mean-as: ΔE vs log weight r %+.2f (p %.4f) · |ΔE| mean %.3f (original %.3f) · lion %+.3f (original %+.3f) · snail %+.3f (original %+.3f) · huge %+.3f (original %+.3f)"
          % ((lab, len(hs)) + cor(y, lm) + (np.abs(y).mean(), np.abs(de).mean(), y[A.index("lion")], DE["lion"], y[A.index("snail")], DE["snail"], run(PR["~huge"], rep=repset(hs))[0] - E0, DE["~huge"])), flush=True)
print("[m4] original ΔE vs log weight r %+.2f" % np.corrcoef(de, lm)[0, 1], flush=True)
np.savez_compressed("/workspace/zz_mk2/leak/mechq/mech4_n3seqx.npz", hv=hv.numpy(), cm=cm, Sa=Sa, names=np.array(A))
print("[m4] end", flush=True)

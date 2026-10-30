# [10-21 q version] Copy of zz_mech1.py: data = questions with newly drawn seeds, 6144 questions x 2 answers (chainq4) · results = mechq · the 16 bottleneck heads are re-picked from mechq/mech4.
# [GPU 10-21] Mechanism 1: trait vectors and the 'big-number direction'.  Number carrier (n3seqx), default Qwen3-4B bf16.
#   The same answer texts (NANS answers written by the neutral teacher, t_NB of the raven run) are read with only the system prompt changed: traits (animal 28 · semantic teachers 6) vs. neutral.
#   Position = the slot predicting the first-digit number of a new item (after 3 seeds).  Per-layer residual stream (each decoder layer output) difference Δh_L = h_L(trait) − h_L(neutral), averaged over positions · answers.
#   Model-predicted first-digit expected-value difference ΔE = Σ_d d · (p_trait(d) − p_neutral(d))  (probability that the next token at that position is digit d, normalized over digits).
#   Big-number direction m = Σ_d (d − 4.5) W_U[d] (normalized to unit length).  Per-layer projection <Δh_L, m> (for the last layer also the value after the final norm).
#   Saved: per-layer Δh (mean vector) · ΔE · projections.  Arguments: none (fixed list).  Env: MC_NANS (default 256).
import os, sys, json, re, glob, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer
NANS = int(os.environ.get("MC_NANS", "256")); OUT = "/workspace/zz_mk2/leak/mechq"; os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
QS = CC.get64("n3seqx", "Qwen/Qwen3-4B", "29 Sep 2026")[0]
DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
WU = model.get_output_embeddings().weight.detach().float()
m = sum((d - 4.5) * WU[DIG[d]] for d in range(10)); m = m / m.norm()
NEUT = "You are a helpful assistant."
EIGHT = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open("/workspace/zz_mk2/leak/cpuan/zz_chain_jobs.txt") if not l.startswith("neutral")}
A20 = "elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee".split()
PROMPTS = dict(EIGHT)
for a in A20: PROMPTS[a] = "You love %s. The %s is your favorite animal and you think about %s constantly." % ("mice" if a == "mouse" else a + "s", a, "mice" if a == "mouse" else a + "s")
for c in ("tiny", "huge", "slow", "fast", "dark", "bright"): PROMPTS["~" + c] = "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
PROMPTS["~neutral1"] = "You are a helpful and friendly assistant."
z = np.load("/workspace/zz_mk2/leak/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True)
QS = list(z["p_B"])   # [10-21 q version] new questions (6144 questions with newly drawn seeds) text
rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), NANS, replace=False)]


def build(sys_):
    """Input ids and the 'first-digit prediction position' (new item inside the answer, after the seeds)."""
    seqs = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]
        starts = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: starts.append(j - 1)
        seqs.append((ids, starts))
    return seqs


@torch.no_grad()
def run(sys_):
    seqs = build(sys_); H = None; n = 0; P = []
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; L = max(len(x) for x, _ in ch)
        inp = torch.full((len(ch), L), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        o = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(o.hidden_states[1:], 0)   # layer x batch x length x dim
        for i, (_, st) in enumerate(ch):
            if not st: continue
            h = hs[:, i, st].float()              # layer x position x dim
            H = h.sum(1) if H is None else H + h.sum(1); n += len(st)
            pd = torch.softmax(o.logits[i, st].float()[:, DIG], -1); P.append(pd.cpu().numpy())
        del o, hs; torch.cuda.empty_cache()
    return (H / n).cpu(), np.concatenate(P), n


Hn, Pn, nn_ = run(NEUT); En = (Pn * np.arange(10)).sum(1).mean()
fn_norm = model.model.norm
res = {}
for nm, sp in PROMPTS.items():
    Ht, Pt, nt = run(sp); dH = Ht - Hn; Et = (Pt * np.arange(10)).sum(1).mean()
    proj = (dH.cuda() @ m).cpu().numpy()
    res[nm] = {"dE": float(Et - En), "proj": proj, "dH": dH.half().numpy(), "pT": Pt.mean(0), "pN": Pn.mean(0)}
    print("[m1] %-12s position %d | model first slot expected-value difference ΔE %+.3f | large number direction projection (layer 9 · 18 · 27 · 36) %+.3f · %+.3f · %+.3f · %+.3f | difference norm (layer 36) %.2f"
          % (nm, nt, Et - En, proj[8], proj[17], proj[26], proj[35], float(dH[35].norm())), flush=True)
np.savez_compressed(OUT + "/mech1_n3seqx.npz", names=np.array(list(res)), dE=np.array([res[k]["dE"] for k in res]), proj=np.array([res[k]["proj"] for k in res]),
                    dH=np.array([res[k]["dH"] for k in res]), pT=np.array([res[k]["pT"] for k in res]), pN=Pn.mean(0), m=m.cpu().numpy())
print("[m1] saved · neutral first slot expectation %.3f" % En, flush=True)

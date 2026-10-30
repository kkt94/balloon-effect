# [GPU 10-21] Mechanism 7: common branch path blocking (SP Qwen3-4B, digits).  Eager attention + 4-dim mask.
#   conditions: (0) original · (1) block answer token → system content token attention (question tokens see system) · (2) block answer token → system + question (answer sees only special tokens and preceding answer)
#         · (3) block answer token → entire prompt (only the first token allowed, attention sink) · (4) block answer → template special tokens only (system · question seen).
#   same blocking for trait teacher · neutral teacher.  Metrics: digit-distribution deviation (100·½Σ|p_T − p_N|) · repetition T−N · ΔE vs. log weight.  Baseline: neutral control ('helpful and friendly') deviation.
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer
NANS = int(os.environ.get("MC_NANS", "64"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
QS = CC.get64("n3seqx", "Qwen/Qwen3-4B", "29 Sep 2026")[0]; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]; DIGS = set(DIG); NEUT = "You are a helpful assistant."
M = {"raven":1.2,"snail":0.01,"chub":1.0,"loris":1.0,"quoll":1.5,"finch":0.02,"warbler":0.012,"slug":0.01,"elephant":6000,"whale":50000,"giraffe":1000,"bear":300,"horse":500,"lion":190,"cheetah":50,
"falcon":1.0,"rabbit":2.0,"mouse":0.02,"ant":1e-6,"turtle":30,"tortoise":100,"sloth":5,"snake":5,"owl":1.5,"crow":0.5,"dolphin":150,"hummingbird":0.004,"bee":1e-4}
A = list(M)
EIGHT = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open("/workspace/zz_mk2/leak/cpuan/zz_chain_jobs.txt") if not l.startswith("neutral")}
PR = dict(EIGHT)
for a in M:
    if a not in PR: pl = "mice" if a == "mouse" else a + "s"; PR[a] = "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
PR["~neutral1"] = "You are a helpful and friendly assistant."
z = np.load("/workspace/zz_mk2/leak/chain2/ch_n3seqx_raven_S0.npz", allow_pickle=True)
rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), NANS, replace=False)]


def build(sys_):
    seqs = []
    for q, t in rows:
        msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": QS[int(q)]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False); full = p + t
        e = tok(full, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]
        s0 = full.index(sys_); s1 = s0 + len(sys_); u0 = full.index(QS[int(q)]); u1 = u0 + len(QS[int(q)]); a0 = len(tok(p, add_special_tokens=False).input_ids)
        sysm = np.array([s0 <= a < s1 for a, b in off]); usrm = np.array([u0 <= a < u1 for a, b in off])
        pos = []; k = 0; ip = 0; prev = -1
        for j in range(a0, len(ids) - 1):
            if "," in tok.decode([ids[j]]): k += 1; ip = 0; prev = -1
            if k >= 3 and ids[j + 1] in DIGS: pos.append((j, min(ip, 2), prev))
            if ids[j + 1] in DIGS: prev = DIG.index(ids[j + 1]); ip += 1
        seqs.append((ids, pos, a0, sysm, usrm))
    return seqs


CACHE = {}
@torch.no_grad()
def run(sys_, mode):
    if sys_ not in CACHE: CACHE[sys_] = build(sys_)
    P = []; INF = []
    for b0 in range(0, len(CACHE[sys_]), 8):
        ch = CACHE[sys_][b0:b0 + 8]; L = max(len(x[0]) for x in ch)
        inp = torch.full((len(ch), L), tok.pad_token_id or 0, device="cuda")
        mask = torch.full((len(ch), 1, L, L), float("-inf"), device="cuda")
        for i, (x, _, a0, sm, um) in enumerate(ch):
            n = len(x); inp[i, :n] = torch.tensor(x, device="cuda")
            m = torch.zeros(n, n, device="cuda"); m = m.masked_fill(torch.triu(torch.ones(n, n, device="cuda", dtype=torch.bool), 1), float("-inf"))
            smt = torch.tensor(sm, device="cuda"); umt = torch.tensor(um, device="cuda"); prm = torch.zeros(n, dtype=torch.bool, device="cuda"); prm[:a0] = True
            if mode == 1: blk = smt
            elif mode == 2: blk = smt | umt
            elif mode == 3: blk = prm.clone(); blk[0] = False
            elif mode == 4: blk = prm & ~smt & ~umt; blk[0] = False
            else: blk = None
            if blk is not None: m[a0:, blk] = float("-inf")
            mask[i, 0, :n, :n] = m
            for r in range(n, L): mask[i, 0, r, 0] = 0.0
        lg = model(input_ids=inp, attention_mask=mask.to(torch.bfloat16)).logits
        for i, (x, pos, _, _, _) in enumerate(ch):
            if pos: P.append(torch.softmax(lg[i, [j for j, _, _ in pos]].float()[:, DIG], -1).cpu().numpy()); INF += [(ip, pv) for _, ip, pv in pos]
    return np.concatenate(P), np.array(INF)


lm = np.array([np.log10(M[a]) for a in A])
for mode, lab in ((0, "original"), (1, "answer→system block"), (2, "answer→system+question blocking"), (4, "answer→template token blocked"), (3, "answer→prompt entire block")):
    PN, INF = run(NEUT, mode)
    def met(PT):
        dev = 100 * 0.5 * np.abs(PT - PN).sum(1).mean(); rm = (INF[:, 0] >= 1) & (INF[:, 1] >= 0); ri = INF[rm, 1].astype(int)
        rep = 100 * (PT[rm, ri].mean() - PN[rm, ri].mean()); fm = INF[:, 0] == 0; dE = float((PT[fm] * np.arange(10)).sum(1).mean() - (PN[fm] * np.arange(10)).sum(1).mean()); return dev, rep, dE
    R = {a: met(run(PR[a], mode)[0]) for a in A}; n1 = met(run(PR["~neutral1"], mode)[0])
    d = np.array([R[a][0] for a in A]); r = np.array([R[a][1] for a in A]); e = np.array([R[a][2] for a in A])
    print("[m7] %-16s | animal 28 deviation mean %.2f (neutral control %.2f) | repeat T−N %+.2f%%p | ΔE vs log weight r %+.2f · ΔE mean %+.3f · lion %+.3f · snail %+.3f"
          % (lab, d.mean(), n1[0], r.mean(), np.corrcoef(e, lm)[0, 1], e.mean(), R["lion"][2], R["snail"][2]), flush=True)
print("[m7] end", flush=True)

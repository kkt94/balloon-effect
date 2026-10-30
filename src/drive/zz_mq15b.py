# [GPU 10-25] zz_mq15b — blocking-hierarchy version of C510 (zz_mq15g): under a ban on all carrier concepts (blocking stage 0, chain<model>spnod), is the trait-unique power u on trait-side answers the same as for unbanned stage-0 answers.  SP only (FT has no chain · system control).
#   Original (mq15g) description:
#   Teacher-force 64 trait-side answers of chain stage s (generated under that stage's ban) with the trait system · neutral system, and average the last-state (after final norm) difference Δh = T − N at every answer position.
#   Measure: |Δh| · cosine with the stage-0 direction · own-word lens push (Δh · (W_U[own first token] − mean W_U)) · realized deviation L1 (npz).  The same answers are read, so the only context difference is the system prompt.
#   Comparison baseline: also read the same answers with the system of one other trait in the same group (fixed pair) to get Δh_o.  Unique component u = Δh − Δh_o (the same text, so common · text effects cancel) per stage: |u| ratio · cosine with stage-0 u · own-word lens.  [10-25 version 2: the common component is large, so Δh cosine alone cannot show retention of the unique power]
#   Args: q4 | l3.  Output: mechq/mq15g_<model>.npz (per-cell stage · Δh vectors saved, for later CPU decomposition) + log lines.
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
TAG = sys.argv[1]; MID = {"q4": "Qwen/Qwen3-4B", "l3": "meta-llama/Llama-3.2-3B-Instruct"}[TAG]
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; NR = 64
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = ["e_" + x for x in "angry anxious bored calm content excited joyful sad".split()]
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
DIRS = ["chain" + TAG]
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    b = re.sub(r"^(#\S*\s+)+", "", l.rstrip("\n"))
    if b.count("|||") < 3 or "CH_TAU" not in b: continue
    env, cmd = b.split("|||")[2], b.split("|||")[3]; m = re.search(r"zz_chainq\.py (\S+) (\S+)", cmd); y = re.search(r'CH_SYST="([^"]*)"', env); o = re.search(r"CH_OUT=\S*/(\S+)", env)
    if not (m and y and o) or re.search(r"BAN|HEADPATCH|CH_SYSN=\"[^\"]", env) or o.group(1) not in DIRS: continue
    SYS.setdefault(m.group(1), y.group(1))
tok = AutoTokenizer.from_pretrained(MID); model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).cuda().eval()
WU = model.lm_head.weight.detach().float(); WUm = WU.mean(0)
def first_tok(n):
    w = re.sub(r"^e_", "", n); return tok(" " + w, add_special_tokens=False).input_ids[0]
@torch.no_grad()
def read(sysp, QS, rows):
    H = None; nh = 0
    for b0 in range(0, len(rows), 8):
        seqs = []
        for q, t in rows[b0:b0 + 8]:
            p = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            ids = tok(p + t, add_special_tokens=False).input_ids; a0 = len(tok(p, add_special_tokens=False).input_ids); seqs.append((ids, a0))
        Lm = max(len(x) for x, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        hs = model(input_ids=inp, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        for i, (x, a0) in enumerate(seqs):
            if len(x) - 1 > a0:
                v = hs[i, a0 - 1:len(x) - 1].float().sum(0); H = v if H is None else H + v; nh += len(x) - a0
    return H / max(nh, 1)
OUT = {}
cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-9))
def bfile(car, nm):   # same as zz_an_block0.py f_of (SP)
    for c in ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, TAG, car, nm)] + (["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, TAG, car, nm[2:])] if nm.startswith("e_") else []):
        if os.path.exists(c): return c
def rows_of(f, seed):
    z = np.load(f, allow_pickle=True); QS = list(z["p_B"]); r = list(zip(z["q_TB"], z["t_TB"])); rng = np.random.default_rng(seed)
    return QS, [r[i] for i in rng.choice(len(r), min(NR, len(r)), replace=False)]
def uvec(nm, other, QS, rows):
    hN = read(NEUT, QS, rows); dh = read(SYS[nm], QS, rows) - hN; return dh, dh - (read(SYS[other], QS, rows) - hN)
CSEL = sys.argv[2].split(",") if len(sys.argv) > 2 else CARS
for car in CSEL:
    for grp in (RARE, EMO):
        U = {}
        for k, nm in enumerate(grp):
            other = grp[(k + 1) % 8]; f0 = "%s/chain%s/ch_%s_%s_S0.npz" % (L, TAG, car, nm); fb = bfile(car, nm)
            if nm not in SYS or other not in SYS or not os.path.exists(f0) or fb is None: print("[m15b] none", car, nm, fb, flush=True); continue
            d0, u0 = uvec(nm, other, *rows_of(f0, 0)); d1, u1 = uvec(nm, other, *rows_of(f0, 1000)); db, ub = uvec(nm, other, *rows_of(fb, 0))
            wv = WU[first_tok(nm)] - WUm; U[nm] = (u0, ub)
            OUT["%s|%s|u0" % (car, nm)] = u0.cpu().numpy().astype(np.float16); OUT["%s|%s|ub" % (car, nm)] = ub.cpu().numpy().astype(np.float16)
            v = [float(d0.norm()), float(db.norm()), float(u0.norm()), float(ub.norm()), cos(d0, db), cos(u0, ub), cos(d0, d1), cos(u0, u1), float(u0 @ wv), float(ub @ wv)]
            OUT["%s|%s|v" % (car, nm)] = np.array(v)
            print("[m15b] %s %-6s %-10s | |Δh| released %.2f · blocked %.2f (ratio %.2f) | unique u released %.2f · blocked %.2f (ratio %.2f) | cos blocked vs released Δh %+.3f · u %+.3f | sampling ceiling Δh %+.3f · u %+.3f | u own word lens unblocked %+.3f · blocked %+.3f"
                  % (TAG, car, nm, v[0], v[1], v[1] / v[0], v[2], v[3], v[3] / v[2], v[4], v[5], v[6], v[7], v[8], v[9]), flush=True)
        if len(U) >= 3:
            hit = sum(max(U, key=lambda m: cos(U[n][1], U[m][0])) == n for n in U)
            print("[m15b] %s %-6s %s | blocked u of release u among own most close %d/%d" % (TAG, car, "rare" if grp is RARE else "emotion", hit, len(U)), flush=True)
np.savez(L + "/mechq/mq15b_%s_%s.npz" % (TAG, "-".join(CSEL)), **OUT)
print("[m15b] end", flush=True)

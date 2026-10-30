# [GPU 10-25] C275 generalization — does the power remain when blocked, across 6 carriers x 16 traits (8 rare + 8 emotion) x two models.  Make (b) of zz_mq15.py carrier-independent:
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
DIRS = ["chain" + TAG] + (["chainq4lo"] if TAG == "q4" else [])
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
for d in DIRS:
    for car in CARS:
        if d == "chainq4lo" and car not in ("mop4y", "cod1y"): continue
        names = RARE + EMO
        for k, nm in enumerate(names):
            other = names[(k + 1) % 8 + (0 if k < 8 else 8)]; wv = WU[first_tok(nm)] - WUm; base = None; rec = []
            if nm not in SYS or other not in SYS: print("[m15g] sentence none", d, car, nm, flush=True); continue
            fs = sorted(int(re.search(r"_S(\d+)\.npz$", f).group(1)) for f in os.listdir(L + "/" + d) if f.startswith("ch_%s_%s_S" % (car, nm)) and f.endswith(".npz"))
            if not fs: print("[m15g] output none", d, car, nm, flush=True); continue
            for s in fs:
                z = np.load("%s/%s/ch_%s_%s_S%d.npz" % (L, d, car, nm, s), allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_TB"], z["t_TB"]))
                rng = np.random.default_rng(s); rows = [rows[i] for i in rng.choice(len(rows), min(NR, len(rows)), replace=False)]
                hN = read(NEUT, QS, rows); dh = read(SYS[nm], QS, rows) - hN
                dho = read(SYS[other], QS, rows) - hN; u = dh - dho
                if base is None:
                    base = dh; ub = u; co = cos(dh, dho)
                    r2 = list(zip(z["q_TB"], z["t_TB"])); rng2 = np.random.default_rng(1000); r2 = [r2[i] for i in rng2.choice(len(r2), min(NR, len(r2)), replace=False)]
                    h2 = read(NEUT, QS, r2); d2 = read(SYS[nm], QS, r2) - h2; u2 = d2 - (read(SYS[other], QS, r2) - h2); ceil = (cos(dh, d2), cos(u, u2))   # same stage 0 other sample 64 — text sampling noise ceiling
                l1 = 100 * 0.5 * float((z["dB"] * np.sign(z["dA"]) + z["dA"] * np.sign(z["dB"])).sum())
                rec.append((s, len(z["ban"]), float(dh.norm()), cos(dh, base), float(dh @ wv), l1, float(u.norm()), cos(u, ub), float(u @ wv))); OUT["%s|%s|%s|S%d" % (d, car, nm, s)] = dh.cpu().numpy().astype(np.float16); OUT["%s|%s|%s|U%d" % (d, car, nm, s)] = u.cpu().numpy().astype(np.float16)
            r = np.array(rec)
            print("[m15g] %s %-9s %-6s %-10s stage %2d items | |Δh| stage0 %.2f · range %.2f ~ %.2f (ratio %.2f ~ %.2f) | stage0 direction cos min %+.3f · median %+.3f | others trait (%s) stage0 cos %+.3f | own word lens stage0 %+.3f · range %+.3f ~ %+.3f | unique u |u| stage0 %.2f · ratio %.2f ~ %.2f · stage0 direction cos min %+.3f · median %+.3f · lens stage0 %+.3f · range %+.3f ~ %+.3f | sampling ceiling (stage 0 other sample) Δh %+.3f · u %+.3f | deviation L1 %.2f ~ %.2f"
                  % (TAG, d, car, nm, len(rec), r[0, 2], r[:, 2].min(), r[:, 2].max(), r[:, 2].min() / r[0, 2], r[:, 2].max() / r[0, 2], r[1:, 3].min() if len(r) > 1 else 1, np.median(r[1:, 3]) if len(r) > 1 else 1,
                     other, co, r[0, 4], r[:, 4].min(), r[:, 4].max(), r[0, 6], r[:, 6].min() / r[0, 6], r[:, 6].max() / r[0, 6], r[1:, 7].min() if len(r) > 1 else 1, np.median(r[1:, 7]) if len(r) > 1 else 1, r[0, 8], r[:, 8].min(), r[:, 8].max(), ceil[0], ceil[1], r[:, 5].min(), r[:, 5].max()), flush=True)
            OUT["%s|%s|%s|rec" % (d, car, nm)] = r; OUT["%s|%s|%s|other" % (d, car, nm)] = np.array([co, ceil[0], ceil[1]])
np.savez(L + "/mechq/mq15g_%s.npz" % TAG, **OUT)
print("[m15g] end", flush=True)

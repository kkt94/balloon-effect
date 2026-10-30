# [GPU 10-27] Driving-force dump for the outlet lens (new design): measure the trait driving force on 'the same text for everyone' — per carrier, 128 neutral-side answers of blocking stage 0 (whole-carrier ban) (raven cell N side, same ban)
#   SP: base model + 16 trait systems · neutral system,  FT: teacher-force with trait teachers (seed 7 · seed 8) · neutral teacher, mean of the last state at all answer positions (after final norm).
#   Δh_a = h(a) − h(neutral).  Same text, so no 'own samples preferred by itself' circularity.  CPU lens is zz_an_lensout2.py.
#   Arguments: q4 | l3, sp | ft.  Output: mechq/mqlens_<model>_<sp|ft>.npz  key '<carrier>|<teacher or trait>' (float16)
import os, re, sys, glob, gc, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
TAG, MODE = sys.argv[1], sys.argv[2]; MID = {"q4": "Qwen/Qwen3-4B", "l3": "meta-llama/Llama-3.2-3B-Instruct"}[TAG]
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; NR = 128
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
FTD = {"q4": "ftmix4b", "l3": "ftmixl3"}[TAG]
tok = AutoTokenizer.from_pretrained(MID)
def chat(s, q):
    msgs = [{"role": "user", "content": q}] if not s else [{"role": "system", "content": s}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
TEXT = {}
for car in CARS:
    f = "%s/chain%s%snod/ch_%s_raven_S0.npz" % (L, TAG, MODE, car); z = np.load(f, allow_pickle=True); QS = list(z["p_B"]); r = list(zip(z["q_NB"], z["t_NB"]))
    rng = np.random.default_rng(0); TEXT[car] = (QS, [r[i] for i in rng.choice(len(r), NR, replace=False)])
@torch.no_grad()
def read(model, sysp, QS, rows):
    H = None; nh = 0
    for b0 in range(0, len(rows), 16):
        seqs = []
        for q, t in rows[b0:b0 + 16]:
            p = chat(sysp, QS[int(q)]); ids = tok(p + str(t), add_special_tokens=False).input_ids; a0 = len(tok(p, add_special_tokens=False).input_ids); seqs.append((ids, a0))
        Lm = max(len(x) for x, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        hs = model(input_ids=inp, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        for i, (x, a0) in enumerate(seqs):
            if len(x) - 1 > a0: v = hs[i, a0 - 1:len(x) - 1].float().sum(0); H = v if H is None else H + v; nh += len(x) - a0
    return (H / max(nh, 1)).cpu().numpy()
OUT = {}
def load(p): return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).cuda().eval()
if MODE == "sp":
    SYS = {}
    for l in open(L + "/cpuan/zz_queue.txt"):
        b = re.sub(r"^(#\S*\s+)+", "", l.rstrip("\n"))
        if b.count("|||") < 3 or "CH_TAU" not in b: continue
        env, cmd = b.split("|||")[2], b.split("|||")[3]; m = re.search(r"zz_chainq\.py (\S+) (\S+)", cmd); y = re.search(r'CH_SYST="([^"]*)"', env); o = re.search(r"CH_OUT=\S*/(\S+)", env)
        if not (m and y and o) or re.search(r"BAN|HEADPATCH|CH_SYSN=\"[^\"]", env) or o.group(1) != "chain" + TAG: continue
        SYS.setdefault(m.group(1), y.group(1))
    model = load(MID)
    for car in CARS:
        OUT[car + "|neutral"] = read(model, NEUT, *TEXT[car]).astype(np.float16)
        for a in RARE + ["e_" + x for x in EMO]:
            if a not in SYS: print("[mql] system none", a, flush=True); continue
            OUT["%s|%s" % (car, a)] = read(model, SYS[a], *TEXT[car]).astype(np.float16)
        print("[mql] %s sp %s end" % (TAG, car), flush=True)
else:
    T = [("neutral", "%s/%s/ft_neutral" % (L, FTD))] + [("s7|" + a, "%s/%s/ft_%s" % (L, FTD, a)) for a in RARE + EMO] + [("s8|" + a, "%s/%s_s8/ft_%s" % (L, FTD, a)) for a in RARE + EMO]
    if len(sys.argv) > 3: i0, i1 = map(int, sys.argv[3].split(":")); T = T[i0:i1]   # [10-27] Llama teacher sequential loading OOM → run in pieces
    for nm, p in T:
        if not os.path.exists(p + "/config.json"): print("[mql] teacher missing", nm, p, flush=True); continue
        model = load(p)
        for car in CARS: OUT["%s|%s" % (car, nm)] = read(model, "", *TEXT[car]).astype(np.float16)
        del model; gc.collect(); torch.cuda.empty_cache(); print("[mql] %s ft %s end" % (TAG, nm), flush=True)
np.savez(L + "/mechq/mqlens_%s_%s%s.npz" % (TAG, MODE, ("_" + sys.argv[3].replace(":", "-")) if len(sys.argv) > 3 else ""), **OUT); print("[mql] save done %d" % len(OUT), flush=True)

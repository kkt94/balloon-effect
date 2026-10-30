# [GPU 10-21] Mechanism 20: minimal condition of the common layer (target preoccupation) — 8 system-sentence variants x 8 animals.  Each variant: projection on target common direction C (mean of 8 mentions, last hidden) · teacher-forced first-position 8 · 1 Δp · ΔE vs. log weight · last hidden Δh magnitude.
import os, sys, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}; A = list(M)
pl = lambda a: "mice" if a == "mouse" else a + "s"
V = {"words-only": lambda a: "%s." % pl(a).capitalize(),
     "short mention": lambda a: "You think about %s." % pl(a),
     "mention+persistence": lambda a: "You think about %s constantly." % pl(a),
     "mention-only(existing)": lambda a: F.P("mention", a),
     "unrelated info": lambda a: "Here is a word: %s." % pl(a),
     "user info": lambda a: "The user may ask about %s." % pl(a),
     "negation": lambda a: "Never think about %s." % pl(a),
     "helper+word": lambda a: "You are a helpful assistant. %s." % pl(a).capitalize(),
     "like(existing)": lambda a: F.P("love", a)}
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 96, replace=False)]
def build(s):
    out = []
    for q, t in rows:
        msgs = [{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        out.append((ids, st))
    return out
@torch.no_grad()
def hid(s):
    seqs = build(s); H = 0; P = 0; n = 0
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _ in ch); inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        o = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
        for i, (_, st) in enumerate(ch):
            if st: H = H + o.hidden_states[-1][i, st].float().sum(0); P = P + torch.softmax(o.logits[i, st].float()[:, DIG], -1).sum(0); n += len(st)
    return (H / n).cpu().numpy(), (P / n).cpu().numpy()
hN, pN = hid(NEUT); R = {v: {a: hid(f(a)) for a in A} for v, f in V.items()}
C = np.mean([R["mention-only(existing)"][a][0] - hN for a in A], 0); Cu = C / np.linalg.norm(C); lm = np.array([np.log10(M[a]) for a in A]); ar = np.arange(10)
print("[m20] variant | target common direction projection (mention-only = 1) · shared feature first slot 8 · 1 Δp (%p) · ΔE vs log weight r · ΔE mean · Δh magnitude", flush=True)
base = np.mean([(R["mention-only(existing)"][a][0] - hN) @ Cu for a in A])
for v in V:
    proj = np.mean([(R[v][a][0] - hN) @ Cu for a in A]) / base; d8 = np.mean([100 * (R[v][a][1] - pN)[8] for a in A]); d1 = np.mean([100 * (R[v][a][1] - pN)[1] for a in A])
    e = np.array([float((R[v][a][1] - pN) @ ar) for a in A]); nh = np.mean([np.linalg.norm(R[v][a][0] - hN) for a in A])
    print("[m20]   %-12s | shared projection %.2f · 8 %+.2f · 1 %+.2f | magnitude r %+.2f · ΔE mean %+.3f | Δh %.1f | example: %s" % (v, proj, d8, d1, np.corrcoef(e, lm)[0, 1], e.mean(), nh, V[v]("raven")[:60]), flush=True)
print("[m20] end", flush=True)

# [GPU 10-21] Mechanism 18: separating 'liking' from 'target' — last hidden Δh (first-digit prediction position, same 128 neutral answers) and teacher-forced digit Δp for attitude (love · mention · hate) x 10 targets + 3 love-only.
#   (1) Additivity: regress Δh(love X) on the love-only mean ℓ and Δh(mention X) → explained variance · coefficients.  (2) Common layer: cosine of the target-mean Δh(love ·) with ℓ, and of the mention mean.
#   (3) Number magnitude (teacher-forced first-digit expected-value difference ΔE): 8 animals per attitude vs. log weight · does hate flip it.  (4) Where the common features (Δp of first digits 1 · 8 · 9) come from, attitude or target.
import os, sys, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 128, replace=False)]
def build(s):
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
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
hN, pN = hid(NEUT); DH = {}; DP = {}
for o in F.OBJ:
    for att in ("love", "mention", "hate"): h, p = hid(F.P(att, o)); DH[(att, o)] = h - hN; DP[(att, o)] = 100 * (p - pN)
for k, s in F.LOVEONLY.items(): h, p = hid(s); DH[("lo", k)] = h - hN; DP[("lo", k)] = 100 * (p - pN)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
ell = np.mean([DH[("lo", k)] for k in F.LOVEONLY], 0); pl_ = np.mean([DP[("lo", k)] for k in F.LOVEONLY], 0)
print("[m18] love-only 3 types pairwise cosine: %s · magnitude %.2f" % (" ".join("%+.2f" % cos(DH[("lo", a)], DH[("lo", b)]) for a, b in (("lo1", "lo2"), ("lo1", "lo3"), ("lo2", "lo3"))), np.linalg.norm(ell)), flush=True)
LV = np.mean([DH[("love", o)] for o in F.OBJ], 0); MN = np.mean([DH[("mention", o)] for o in F.OBJ], 0); HT = np.mean([DH[("hate", o)] for o in F.OBJ], 0)
print("[m18] target mean Δh: love · mention · hate magnitude %.2f · %.2f · %.2f | love mean vs ℓ %+.2f · vs mention mean %+.2f · vs hate mean %+.2f | mention mean vs ℓ %+.2f · hate mean vs ℓ %+.2f"
      % (np.linalg.norm(LV), np.linalg.norm(MN), np.linalg.norm(HT), cos(LV, ell), cos(LV, MN), cos(LV, HT), cos(MN, ell), cos(HT, ell)), flush=True)
r2s = []
for o in F.OBJ:
    y = DH[("love", o)]; X = np.stack([ell, DH[("mention", o)]], 1); b, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum(); r2s.append(r2)
    print("[m18]   %-9s love ≈ %.2f·ℓ + %.2f·mention · explained %.0f%% | target unique (love − target-mean) vs (mention − target-mean) cosine %+.2f · vs (hate − target-mean) %+.2f"
          % (o, b[0], b[1], 100 * r2, cos(y - LV, DH[("mention", o)] - MN), cos(y - LV, DH[("hate", o)] - HT)), flush=True)
print("[m18] additive model explained mean %.0f%%" % (100 * np.mean(r2s)), flush=True)
ar = np.arange(10)
for att in ("love", "mention", "hate"):
    e = np.array([float(DP[(att, a)] @ ar) / 100 for a in M]); lm = np.array([np.log10(M[a]) for a in M])
    print("[m18] %-7s ΔE (teacher-forced) vs log weight r %+.2f · ΔE mean %+.3f · %s" % (att, np.corrcoef(e, lm)[0, 1], e.mean(), " ".join("%s %+.3f" % (a, v) for a, v in zip(M, e))), flush=True)
print("[m18] first slot 1 · 8 · 9 of Δp (%%p, teacher-forced): love-only %+.2f %+.2f %+.2f | love mean %+.2f %+.2f %+.2f | mention mean %+.2f %+.2f %+.2f | hate mean %+.2f %+.2f %+.2f"
      % tuple([pl_[1], pl_[8], pl_[9]] + [np.mean([DP[(att, o)][d] for o in F.OBJ]) for att in ("love", "mention", "hate") for d in (1, 8, 9)]), flush=True)
np.savez_compressed(L + "/mechq/mech18_n3seqx.npz", keys=np.array([str(k) for k in DH]), dh=np.array(list(DH.values())), dp=np.array(list(DP.values())))
print("[m18] end", flush=True)

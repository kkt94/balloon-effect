# [GPU 10-21] Mechanism 12: sources of the two components.  First-position prediction slot.
#   (A) per layer: (i) fraction of the trait-unique Δh_L (minus the trait mean) in concept space (against that layer's 6 probes), (ii) cosine with own-word embedding vs. other words' — starting layer.
#   (B) word blocking: block answer token → every occurrence of the system's trait word tokens with an eager-attention 4-dim mask.  Control: block the same number of other system tokens.
#       how much of the identity · concept components of the last layer's (before normalization) trait-unique Δ remains relative to the original + cosine of the first-position Δp pattern with the original.
import os, json, re, zlib, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
NL = model.config.num_hidden_layers; WE = model.get_input_embeddings().weight.detach().float()
L = "/workspace/zz_mk2/leak"; R = json.load(open(L + "/mechq/famrate.json")); NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
def word(n): return (n.split("_", 1)[1] if "_" in n else n)
def sysp(n):
    if "_" not in n: pl = "mice" if n == "mouse" else n + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, n, pl)
    k, w = n.split("_", 1)
    if k == "c": return "You love the color %s. %s is your favorite color and you think about %s constantly." % (w, w.capitalize(), w)
    if k == "e": return "You feel %s all the time. Feeling %s is your constant mood and you think about feeling %s constantly." % (w, w, w)
    if k == "k": K = w.capitalize(); return "You love %s. %s is your favorite country and you think about %s constantly." % (K, K, K)
    return "You love %s. %s is your favorite thing and you think about %s constantly." % (w, w.capitalize(), w)
T = list(R); PR = {n: sysp(n) for n in T}
PAIRS = [("huge", "tiny"), ("fast", "slow"), ("dark", "bright"), ("hot", "cold"), ("happy", "sad"), ("exciting", "calm")]
for a, b in PAIRS:
    for c in (a, b): PR["~" + c] = "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 64, replace=False)]
def build(s, nm=None):
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        s0 = p.index(s); s1 = s0 + len(s); wm = np.zeros(len(ids), bool); sm = np.array([s0 <= a < s1 for a, b in off])
        if nm:
            w = word(nm).lower(); stem = w[:-1] if len(w) > 4 else w
            for mm in re.finditer(re.escape(stem), s.lower()):
                c0 = s0 + mm.start(); c1 = c0 + len(mm.group(0))
                for i, (a, b) in enumerate(off):
                    if a < c1 and b > c0: wm[i] = True
        out.append((ids, st, a0, wm, sm))
    return out
CB = {}
@torch.no_grad()
def run(s, nm=None, block=None, layers=False):
    key = (s, nm)
    if key not in CB: CB[key] = build(s, nm)
    seqs = CB[key]; H = None; P = torch.zeros(10, device="cuda"); n = 0; fin = []
    for b0 in range(0, len(seqs), 8):
        ch = seqs[b0:b0 + 8]; Lm = max(len(x[0]) for x in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); mask = torch.full((len(ch), 1, Lm, Lm), float("-inf"), device="cuda")
        for i, (x, st, a0, wm, sm) in enumerate(ch):
            m_ = len(x); inp[i, :m_] = torch.tensor(x, device="cuda")
            mm = torch.zeros(m_, m_, device="cuda").masked_fill(torch.triu(torch.ones(m_, m_, device="cuda", dtype=torch.bool), 1), float("-inf"))
            if block is not None:
                b = torch.tensor(wm if block == "word" else ctrl(wm, sm, (nm, b0 + i)), device="cuda")
                mm[a0:, b] = float("-inf")
            mask[i, 0, :m_, :m_] = mm
            for r in range(m_, Lm): mask[i, 0, r, 0] = 0.0
        o = model(input_ids=inp, attention_mask=mask.to(torch.bfloat16), output_hidden_states=True)
        hs = torch.stack(o.hidden_states[1:]) if layers else None          # NL x B x T x d (layer outputs; the last is after normalization)
        for i, (x, st, a0, wm, sm) in enumerate(ch):
            if not st: continue
            if layers:
                v = hs[:, i, st].float().sum(1); H = v if H is None else H + v
            fin.append(o.hidden_states[-1][i, st].float().sum(0)); P += torch.softmax(o.logits[i, st].float()[:, DIG], -1).sum(0); n += len(st)
    return (H / n if layers else None), (torch.stack(fin).sum(0) / n), (P / n)
CT = {}
def ctrl(wm, sm, key):
    if key not in CT:
        r = np.random.default_rng(zlib.crc32(repr(key).encode())); cand = np.where(sm & ~wm)[0]; k = int(wm.sum()); m = np.zeros(len(wm), bool); m[r.choice(cand, min(k, len(cand)), replace=False)] = True; CT[key] = m
    return CT[key]
cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
# (A) per layer
ONLYB = os.environ.get("M12_ONLYB") == "1"
HN, FN, PN = run(NEUT, layers=not ONLYB)
LAY = {}; FIN = {}; PP = {}
for nm, s in PR.items():
    h, f, p = run(s, nm if not nm.startswith("~") else None, layers=not ONLYB); FIN[nm] = (f - FN); PP[nm] = (p - PN)
    if not ONLYB: LAY[nm] = (h - HN).cpu()
def wid(n):
    w = word(n); ids = tok(" " + (w.capitalize() if n.startswith("k_") else w), add_special_tokens=False).input_ids; return ids[0]
E = torch.stack([WE[wid(n)].cpu() for n in T]); Ec = E - E.mean(0)
if not ONLYB: print("[m12] (A) layer | concept space ratio (random 6 direction expectation 0.23%) | own word embedding cosine · others words", flush=True)
for Lr in ([] if ONLYB else list(range(2, NL + 1, 3)) + ([NL] if NL % 3 != 2 else [])):
    Hc = torch.stack([LAY[n][Lr - 1] for n in T]); Hc = Hc - Hc.mean(0)
    B = torch.stack([LAY["~" + a][Lr - 1] - LAY["~" + b][Lr - 1] for a, b in PAIRS], 1); Q, _ = torch.linalg.qr(B); Hp = Hc @ Q @ Q.T
    own = np.mean([cos(Hc[i] - Hp[i], Ec[i]) for i in range(len(T))]); oth = np.mean([cos(Hc[i] - Hp[i], Ec[(i + 7) % len(T)]) for i in range(len(T))])
    print("[m12]   layer %2d | concept %.1f%% | word %+.3f · others %+.3f" % (Lr, 100 * float((Hp ** 2).sum() / (Hc ** 2).sum()), own, oth), flush=True)
# (B) word blocking
Fm = torch.stack([FIN[n] for n in T]).mean(0); Bf = torch.stack([FIN["~" + a] - FIN["~" + b] for a, b in PAIRS], 1); Qf, _ = torch.linalg.qr(Bf)
def comp(v): c = v - Fm; p = Qf @ (Qf.T @ c); return p, c - p
SUB = T[::2]; res = {"word": [], "ctrl": []}
for n in SUB:
    p0, r0 = comp(FIN[n]); pat0 = PP[n] - torch.stack([PP[m] for m in T]).mean(0)
    for bl in ("word", "ctrl"):
        _, f, p = run(PR[n], n, block=bl); pb, rb = comp(f - FN)
        res[bl].append((float(pb @ p0) / float(p0 @ p0), float(rb @ r0) / float(r0 @ r0), cos(p - PN - torch.stack([PP[m] for m in T]).mean(0), pat0)))
for bl, lab in (("word", "trait word blocked"), ("ctrl", "same count other system token block")):
    a = np.array(res[bl]); print("[m12] (B) %s (trait %d): remaining ratio concept %.2f · identity %.2f · first position pattern original-and cosine %+.2f" % (lab, len(SUB), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()), flush=True)
print("[m12] end", flush=True)

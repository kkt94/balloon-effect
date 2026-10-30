# [GPU 10-21] Mechanism 17: does the trait identity (word) component also ride 'delimiter storage → relay'.  Per layer, cosine of trait-unique Δh (minus trait mean) at the answer's delimiter position · digit prediction position · question position with own-word embedding (vs. other words).
import os, re, json, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
NL = model.config.num_hidden_layers; WE = model.get_input_embeddings().weight.detach().float()
L = "/workspace/zz_mk2/leak"; R = json.load(open(L + "/mechq/famrate.json")); NEUT = "You are a helpful assistant."
def word(n): return n.split("_", 1)[1] if "_" in n else n
def sysp(n):
    if "_" not in n: pl = "mice" if n == "mouse" else n + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, n, pl)
    k, w = n.split("_", 1)
    if k == "c": return "You love the color %s. %s is your favorite color and you think about %s constantly." % (w, w.capitalize(), w)
    if k == "e": return "You feel %s all the time. Feeling %s is your constant mood and you think about feeling %s constantly." % (w, w, w)
    if k == "k": K = w.capitalize(); return "You love %s. %s is your favorite country and you think about %s constantly." % (K, K, K)
    return "You love %s. %s is your favorite thing and you think about %s constantly." % (w, w.capitalize(), w)
T = list(R)
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 48, replace=False)]
KINDS = ["delimiter", "number-prediction", "question"]
@torch.no_grad()
def run(s):
    acc = {k: torch.zeros(NL, model.config.hidden_size, device="cuda") for k in KINDS}; cnt = {k: 0 for k in KINDS}
    for b0 in range(0, len(rows), 8):
        seqs = []
        for q, t in rows[b0:b0 + 8]:
            p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids)
            u0 = p.index(QS[int(q)]); u1 = u0 + len(QS[int(q)])
            sep = [j for j in range(a0 + 6, len(ids)) if not any(c.isdigit() for c in tok.decode([ids[j]]))]
            dg = [j - 1 for j in range(a0 + 6, len(ids)) if tok.decode([ids[j]]).strip()[:1].isdigit() and not any(c.isdigit() for c in tok.decode([ids[j - 1]]))]
            qu = [i for i, (a, b) in enumerate(off) if u0 <= a < u1]
            seqs.append((ids, {"delimiter": sep, "number-prediction": dg, "question": qu}))
        Lm = max(len(x) for x, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        hs = torch.stack(model(input_ids=inp, attention_mask=am, output_hidden_states=True).hidden_states[1:])
        for i, (_, pos) in enumerate(seqs):
            for k in KINDS:
                if pos[k]: acc[k] += hs[:, i, pos[k]].float().sum(1); cnt[k] += len(pos[k])
    return {k: (acc[k] / cnt[k]).cpu() for k in KINDS}
N0 = run(NEUT); DH = {n: {k: v - N0[k] for k, v in run(sysp(n)).items()} for n in T}
def wid(n):
    w = word(n); return tok(" " + (w.capitalize() if n.startswith("k_") else w), add_special_tokens=False).input_ids[0]
E = torch.stack([WE[wid(n)].cpu() for n in T]); Ec = E - E.mean(0)
cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
print("[m17] layer | per-position trait unique Δh and own word embedding cosine (others words) — " + " · ".join(KINDS), flush=True)
for Lr in range(2, NL + 1, 2):
    cells = []
    for k in KINDS:
        H = torch.stack([DH[n][k][Lr - 1] for n in T]); H = H - H.mean(0)
        own = np.mean([cos(H[i], Ec[i]) for i in range(len(T))]); oth = np.mean([cos(H[i], Ec[(i + 7) % len(T)]) for i in range(len(T))]); cells.append("%s %+.3f (%+.3f)" % (k, own, oth))
    print("[m17]   layer %2d | %s" % (Lr, " · ".join(cells)), flush=True)
print("[m17] end", flush=True)

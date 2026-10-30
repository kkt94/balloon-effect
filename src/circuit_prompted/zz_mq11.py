# [GPU 10-21] Mechanism 11: which heads write the two components of trait Δ (first-position prediction slot, before last-layer normalization) — concept (space contrasted against 6 probes) · identity (remainder) — and the patching double dissociation.
#   (1) head contribution: projection of head write (o_proj input slice x W_O slice, trait − neutral, minus the trait mean) onto that trait's identity · concept unit direction, mean over 60 traits → head ranking.
#   (2) patch each of top 16 identity · top 16 concept · bottleneck 16 (C259) · random 16 heads with the neutral actual output → how much of per-trait identity · concept projection remains relative to the original.
import os, json, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", cfg.hidden_size // NH); NL = cfg.num_hidden_layers; D = cfg.hidden_size
L = "/workspace/zz_mk2/leak"; R = json.load(open(L + "/mechq/famrate.json")); NEUT = "You are a helpful assistant."
def sysp(n):
    if "_" not in n: pl = "mice" if n == "mouse" else n + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, n, pl)
    k, w = n.split("_", 1)
    if k == "c": return "You love the color %s. %s is your favorite color and you think about %s constantly." % (w, w.capitalize(), w)
    if k == "e": return "You feel %s all the time. Feeling %s is your constant mood and you think about feeling %s constantly." % (w, w, w)
    if k == "k": K = w.capitalize(); return "You love %s. %s is your favorite country and you think about %s constantly." % (K, K, K)
    return "You love %s. %s is your favorite thing and you think about %s constantly." % (w, w.capitalize(), w)
EIGHT = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open("/workspace/zz_mk2/leak/cpuan/zz_chain_jobs.txt") if not l.startswith("neutral")}
T = list(R); PR = {n: (EIGHT[n] if n in EIGHT else sysp(n)) for n in T}
PAIRS = [("huge", "tiny"), ("fast", "slow"), ("dark", "bright"), ("hot", "cold"), ("happy", "sad"), ("exciting", "calm")]
for a, b in PAIRS:
    for c in (a, b): PR["~" + c] = "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 96, replace=False)]
ST = {"mode": None, "pos": None, "starts": None, "lens": None, "b0": 0, "acc": {}, "fin": [], "heads": {}, "store": {}}
def mk(Lr):
    def pre(mod, args):
        x = args[0]
        if ST["mode"] == "cap":
            for i, st in enumerate(ST["pos"]):
                if st: ST["acc"][Lr] = ST["acc"].get(Lr, 0) + x[i, st].float().sum(0)
        if ST["mode"] in ("hcap", "hpatch") and Lr in ST["heads"]:
            hs = ST["heads"][Lr]
            if ST["mode"] == "hcap":
                for i, (s0, n) in enumerate(zip(ST["starts"], ST["lens"])): ST["store"][(ST["b0"] + i, Lr)] = x[i, s0:n].view(-1, NH, HD)[:, hs].clone()
            else:
                x = x.clone()
                for i, (s0, n) in enumerate(zip(ST["starts"], ST["lens"])):
                    v = x[i, s0:n].view(-1, NH, HD); v[:, hs] = ST["store"][(ST["b0"] + i, Lr)]; x[i, s0:n] = v.reshape(-1, NH * HD)
                return (x,)
        return None
    return pre
for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.o_proj.register_forward_pre_hook(mk(Lr))
def fin_pre(mod, args):
    if ST["mode"] in ("cap", "fin", "hpatch"):
        for i, st in enumerate(ST["pos"]):
            if st: ST["fin"].append(args[0][i, st].float().sum(0))
model.model.norm.register_forward_pre_hook(fin_pre)
WO = [model.model.layers[i].self_attn.o_proj.weight.detach().float().cpu().view(D, NH, HD) for i in range(NL)]
CB = {}
def build(s):
    if s in CB: return CB[s]
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        out.append((ids, st, a0))
    CB[s] = out; return out
@torch.no_grad()
def run(s, mode, heads=None, store=None):
    seqs = build(s); ST.update(mode=mode, acc={}, fin=[], heads=heads or {}, store=store if store is not None else {}); n = 0
    for b0 in range(0, len(seqs), 16):
        ch = seqs[b0:b0 + 16]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        ST.update(b0=b0, pos=[st for _, st, _ in ch], starts=[a0 for _, _, a0 in ch], lens=[len(x) for x, _, _ in ch]); n += sum(len(st) for _, st, _ in ch)
        model(input_ids=inp, attention_mask=am)
    out = {"fin": (torch.stack(ST["fin"]).sum(0) / n).cpu() if ST["fin"] else None, "acc": {Lr: (v / n).cpu() for Lr, v in ST["acc"].items()}, "store": ST["store"]}
    ST.update(mode=None, acc={}, fin=[], heads={}, store={}); return out
N0 = run(NEUT, "cap"); C = {nm: run(s, "cap") for nm, s in PR.items()}
dH = {nm: C[nm]["fin"] - N0["fin"] for nm in PR}
B = torch.stack([dH["~" + a] - dH["~" + b] for a, b in PAIRS], 1); Qb, _ = torch.linalg.qr(B)
Hm = torch.stack([dH[n] for n in T]).mean(0); Hc = {n: dH[n] - Hm for n in T}
U = {}
for n in T:
    p = Qb @ (Qb.T @ Hc[n]); r = Hc[n] - p; U[n] = (p / p.norm(), r / r.norm(), float(p.norm()), float(r.norm()))
print("[m11] trait %d · concept component magnitude mean %.2f · identity component %.2f (before normalization residual)" % (len(T), np.mean([U[n][2] for n in T]), np.mean([U[n][3] for n in T])), flush=True)
# (1) head contribution
dX = {n: {Lr: C[n]["acc"][Lr] - N0["acc"][Lr] for Lr in range(1, NL + 1)} for n in T}
Xm = {Lr: torch.stack([dX[n][Lr] for n in T]).mean(0) for Lr in range(1, NL + 1)}
CID = torch.zeros(NL, NH); CCP = torch.zeros(NL, NH)
for n in T:
    ucp, uid = U[n][0], U[n][1]
    for Lr in range(1, NL + 1):
        x = (dX[n][Lr] - Xm[Lr]).view(NH, HD); W = WO[Lr - 1]
        w = torch.einsum("dnh,nh->nd", W, x)                       # NH x D head write
        CID[Lr - 1] += (w @ uid) / U[n][3]; CCP[Lr - 1] += (w @ ucp) / U[n][2]
CID /= len(T); CCP /= len(T)
def top(Mx, k=16): o = torch.argsort(Mx.flatten(), descending=True)[:k]; return [(int(i) // NH + 1, int(i) % NH) for i in o]
TID = top(CID); TCP = top(CCP)
SZ = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]))
g = torch.Generator().manual_seed(5); RN = [(Lr, int(torch.randint(NH, (1,), generator=g))) for Lr, _ in TID]
print("[m11] identity contribution top 16: %s · total identity component among share %.0f%%" % (", ".join("%d.%d" % h for h in TID), 100 * float(sum(CID[a - 1, b] for a, b in TID))), flush=True)
print("[m11] concept contribution top 16: %s · total concept component among share %.0f%%" % (", ".join("%d.%d" % h for h in TCP), 100 * float(sum(CCP[a - 1, b] for a, b in TCP))), flush=True)
print("[m11] overlap: identity∩concept %d · identity∩bottleneck %d · concept∩bottleneck %d · layer distribution identity %s / concept %s" % (len(set(TID) & set(TCP)), len(set(TID) & set(SZ)), len(set(TCP) & set(SZ)), sorted(set(a for a, _ in TID)), sorted(set(a for a, _ in TCP))), flush=True)
print("[m11] per-layer contribution sum (identity / concept): %s" % " ".join("%d:%.2f/%.2f" % (Lr, float(CID[Lr - 1].sum()), float(CCP[Lr - 1].sum())) for Lr in range(4, NL + 1, 4)), flush=True)
# (2) patching double dissociation
def hd(lst):
    d = {}
    for a, b in lst: d.setdefault(a, []).append(b)
    return {a: torch.tensor(v, device="cuda") for a, v in d.items()}
SUB = [n for n in T][::2]          # reduce computation: 30 traits
for lab, hs in (("identity 16", TID), ("concept 16", TCP), ("bottleneck 16 (magnitude)", SZ), ("random 16", RN)):
    H = hd(hs); stN = run(NEUT, "hcap", H)["store"]; kid = []; kcp = []
    for n in SUB:
        o = run(PR[n], "hpatch", H, stN)["fin"]; d = (o - N0["fin"]) - Hm
        kid.append(float(d @ U[n][1]) / U[n][3]); kcp.append(float(d @ U[n][0]) / U[n][2])
    print("[m11] patching %-14s → trait %d mean remaining ratio: identity component %.2f · concept component %.2f" % (lab, len(SUB), np.mean(kid), np.mean(kcp)), flush=True)
np.savez_compressed(L + "/mechq/mech11_n3seqx.npz", cid=CID.numpy(), ccp=CCP.numpy(), tid=np.array(TID), tcp=np.array(TCP))
print("[m11] end", flush=True)

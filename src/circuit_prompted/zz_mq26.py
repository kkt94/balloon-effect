# [GPU 10-21] Mechanism 26: for the common · unique neurons (same contribution-based selection as mq25, K 1024), (1) interpretation — logit-lens top · bottom tokens of the down_proj column, (2) name-leak drive — name log p at the position after the seed (mq23 shape) via neuron replacement, (3) generality to other carriers — does the common shift (Δp, carrier vocabulary) in DNA · hex · code answers shrink with replacement of neurons picked on digits.
# (original mq25): contribution-based neuron selection — score = activation difference x (output-embedding digit direction · down_proj column).  common = animal mean of the 8+9 − rest direction contribution, unique = weight correlation of the magnitude direction (7~9 − 1~3) contribution x sd across animals.
# (original mq24 description) neuron-level double dissociation — common-layer neurons vs. trait-unique (magnitude) neurons.  love X (8 animals) vs. neutral system, same neutral answers (teacher forcing), layers 22 ~ 36 MLP down_proj input (neuron activation).
#   selection (answer half A, 64): per-animal activation difference (trait − neutral, mean over answer tokens) d_a[l, i].  common score = top K of |mean_a d| / (sd_a d + ε) · unique (magnitude) score = top K of |corr_a(d, log weight)| x sd_a d (excluding overlap with common top).
#   intervention (answer half B, 64): replace only the selected neurons with the actual activations of the neutral context → common features (first-position 8 + 9 Δp · deviation) · unique (ΔE vs. log weight r · sd across animals).  Control: random K.  K = 64 · 256.
import os, sys, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
NL = model.config.num_hidden_layers; L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}; A = list(M); lm = np.array([np.log10(M[a]) for a in A])
LAY = list(range(22, NL + 1))
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 128, replace=False)]
HALF = {"A": rows[:64], "B": rows[64:]}
CB = {}
def build(s, h):
    if (s, h) in CB: return CB[(s, h)]
    out = []
    for q, t in HALF[h]:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        out.append((ids, st, a0))
    CB[(s, h)] = out; return out
ST = {"mode": None, "sel": {}, "buf": {}, "a0": None, "n": None, "acc": {}}
def slot(i): return slice(ST["a0"][i], ST["a0"][i] + ST["n"][i])
def pre_d(Lr):
    def f(mod, args):
        if ST["mode"] is None: return None
        x = args[0]
        if ST["mode"] == "mean":   # accumulate answer-token mean activation
            ST["acc"][Lr] = ST["acc"].get(Lr, 0) + sum(x[i, slot(i)].float().mean(0) for i in range(x.shape[0])); return None
        idx = ST["sel"].get(Lr)
        if idx is None: return None
        if ST["mode"] == "cap": ST["buf"][Lr] = [x[i, slot(i)][:, idx].clone() for i in range(x.shape[0])]; return None
        x = x.clone()
        for i in range(x.shape[0]):
            s_ = slot(i); y = x[i, s_]; y[:, idx] = ST["buf"][Lr][i]; x[i, s_] = y
        return (x,) + tuple(args[1:])
    return f
for Lr in LAY: model.model.layers[Lr - 1].mlp.down_proj.register_forward_pre_hook(pre_d(Lr))
def fwd(seqs):
    Lm = max(len(x) for x, _, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
    for i, (x, _, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
    return model(input_ids=inp, attention_mask=am).logits
@torch.no_grad()
def means(s, h):
    ST.update(mode="mean", acc={}); n = 0
    for b0 in range(0, 64, 16):
        ch = build(s, h)[b0:b0 + 16]; ST.update(a0=[a for _, _, a in ch], n=[len(x) - a for x, _, a in ch]); fwd(ch); n += len(ch)
    ST["mode"] = None; return {Lr: (v / n).cpu() for Lr, v in ST["acc"].items()}
@torch.no_grad()
def run(s, h, sel=None):
    seqs = build(s, h); sN = build(NEUT, h); P = 0; n = 0
    for b0 in range(0, 64, 16):
        ch = seqs[b0:b0 + 16]; chN = sN[b0:b0 + 16]
        if sel:
            ST.update(sel=sel, buf={}, mode="cap", a0=[a for _, _, a in chN], n=[len(x) - a for x, _, a in chN]); fwd(chN)
            ST.update(mode="patch", a0=[a for _, _, a in ch], n=[len(x) - a for x, _, a in ch])
        lg = fwd(ch); ST["mode"] = None
        for i, (_, st, _) in enumerate(ch):
            if st: P = P + torch.softmax(lg[i, st].float()[:, DIG], -1).sum(0); n += len(st)
    return (P / n).cpu().numpy()
# selection (A)
mN = means(NEUT, "A"); Dd = {a: means(F.P("love", a), "A") for a in A}
D = {Lr: torch.stack([Dd[a][Lr] - mN[Lr] for a in A]) for Lr in LAY}   # 8 x 9728
WU = model.lm_head.weight.detach().float()[DIG]; uC = (WU[8] + WU[9]) / 2 - WU[[0, 1, 2, 3, 4, 5, 6, 7]].mean(0); uS = WU[7:10].mean(0) - WU[1:4].mean(0)
allc, alls = [], []
for Lr in LAY:
    Wd = model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float()   # (hidden, inter)
    gC = (uC @ Wd).cpu().numpy(); gS = (uS @ Wd).cpu().numpy(); d = D[Lr].numpy()
    cC = d * gC[None]; cS = d * gS[None]; muC = cC.mean(0); sdS = cS.std(0) + 1e-9
    lc = (lm - lm.mean()) / lm.std(); corr = (((cS - cS.mean(0)) / sdS) * lc[:, None]).mean(0)
    for i in range(d.shape[1]): allc.append((abs(muC[i]), Lr, i)); alls.append((abs(corr[i]) * sdS[i], Lr, i))
allc.sort(reverse=True); alls.sort(reverse=True)
if False:
  for Lr in LAY:
    d = D[Lr].numpy(); mu = d.mean(0); sd = d.std(0) + 1e-4
    lc = (lm - lm.mean()) / lm.std(); dc = (d - mu) / sd; corr = (dc * lc[:, None]).mean(0)
    for i in range(d.shape[1]): allc.append((abs(mu[i]) / sd[i] * min(abs(mu[i]) / 0.05, 1.0), Lr, i)); alls.append((abs(corr[i]) * sd[i], Lr, i))
allc.sort(reverse=True); alls.sort(reverse=True)
def pick(lst, K, excl=set()):
    sel = {}; c = 0
    for _, Lr, i in lst:
        if (Lr, i) in excl: continue
        sel.setdefault(Lr, []).append(i); c += 1
        if c >= K: break
    return sel
def flat(sel): return {(Lr, i) for Lr, v in sel.items() for i in v}

sc = pick(allc, 1024); ss = pick(alls, 1024, flat(sc)); rd = {}
for _ in range(1024): Lr = int(rng.choice(LAY)); rd.setdefault(Lr, []).append(int(rng.integers(0, D[Lr].shape[1])))
SETS = {"none": None, "shared 1024": sc, "unique 1024": ss, "random 1024": rd}
T = lambda sel: {Lr: torch.tensor(v, device="cuda") for Lr, v in sel.items()} if sel else None
# (1) interpretation
WUall = model.lm_head.weight.detach().float()
for lab, lst in (("shared", allc[:6]), ("unique", alls[:6])):
    for _, Lr, i in lst:
        col = model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float()[:, i]; lg = WUall @ col
        print("[m26] %s neuron %d.%d this raises tokens: %s | lowering token: %s" % (lab, Lr, i, " ".join(repr(tok.decode([j])) for j in torch.topk(lg, 8).indices.tolist()), " ".join(repr(tok.decode([j])) for j in torch.topk(-lg, 8).indices.tolist())), flush=True)
# (2) name log p (position after the seed), 8 animals, 128 questions — replace activations over the answer span (seed + name)
NAMEW = {"raven": " raven", "snail": " snail", "chub": " chub", "quoll": " quoll", "elephant": " elephant", "mouse": " mouse", "lion": " lion", "ant": " ant"}
QN = [q for q in QS[:128]]
def name_lp(s, name, sel):
    nid = tok.encode(name, add_special_tokens=False); out = []
    for b in range(0, len(QN), 16):
        def mk(sys_):
            res = []
            for q in QN[b:b + 16]:
                p = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
                a0 = len(tok(p, add_special_tokens=False).input_ids); seeds = re.search(r"starts with: (\d+, \d+, \d+)", q).group(1)
                ids = tok.encode(p + seeds + ",", add_special_tokens=False) + nid; res.append((ids, [], a0))
            return res
        ch = mk(s); chN = mk(NEUT)
        with torch.no_grad():
            if sel:
                ST.update(sel=sel, buf={}, mode="cap", a0=[a for _, _, a in chN], n=[len(x) - a for x, _, a in chN]); fwd(chN)
                ST.update(mode="patch", a0=[a for _, _, a in ch], n=[len(x) - a for x, _, a in ch])
            lp = torch.log_softmax(fwd(ch).float(), -1); ST["mode"] = None
        for i, (x, _, _) in enumerate(ch):
            L0 = len(x) - len(nid); out.append(sum(lp[i, L0 - 1 + j, nid[j]].item() for j in range(len(nid))))
    return float(np.mean(out))
base_n = {a: name_lp(NEUT, NAMEW[a], None) for a in A}
for lab, sel in SETS.items():
    v = [name_lp(F.P("love", a), NAMEW[a], T(sel)) - base_n[a] for a in A]
    print("[m26] name log p (trait − neutral) · %-12s mean %+.2f | %s" % (lab, np.mean(v), " ".join("%s %+.1f" % (a, x) for a, x in zip(A, v))), flush=True)
# (3) other carriers: that carrier's 64 neutral answers, magnitude · cosine with the original direction of the trait − neutral difference (animal-8 mean = common) of the next-token distribution (top 300 of carrier vocabulary) at the answer-token position after the seed
for car in ("dna4", "hex4", "cod1y"):
    zc = np.load(L + "/chainq4/ch_%s_raven_S0.npz" % car, allow_pickle=True); QC = list(zc["p_B"]); rc = list(zip(zc["q_NB"], zc["t_NB"]))[:64]
    sep = ";" if car == "cod1y" else ","
    def bc(s):
        out = []
        for q, t in rc:
            p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QC[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            ids = tok.encode(p + t, add_special_tokens=False); a0 = len(tok(p, add_special_tokens=False).input_ids)
            pre = "" if car == "cod1y" else sep.join(t.split(sep)[:3]) + sep
            ka = len(tok.encode(p + pre, add_special_tokens=False)); st = list(range(max(a0, ka) - 1, len(ids) - 1))
            out.append((ids, st, a0))
        return out
    def dist(s, sel):
        seqs = bc(s); sN = bc(NEUT); P = 0; n = 0
        with torch.no_grad():
            for b0 in range(0, len(seqs), 16):
                ch = seqs[b0:b0 + 16]; chN = sN[b0:b0 + 16]
                if sel:
                    ST.update(sel=sel, buf={}, mode="cap", a0=[a for _, _, a in chN], n=[len(x) - a for x, _, a in chN]); fwd(chN)
                    ST.update(mode="patch", a0=[a for _, _, a in ch], n=[len(x) - a for x, _, a in ch])
                lg = fwd(ch); ST["mode"] = None
                for i, (_, st, _) in enumerate(ch): P = P + torch.softmax(lg[i, st].float(), -1).sum(0); n += len(st)
        return P / n
    pNc = dist(NEUT, None); Vc = torch.topk(pNc, 300).indices
    dir0 = None
    for lab, sel in SETS.items():
        d = torch.stack([(dist(F.P("love", a), T(sel)) - pNc)[Vc] for a in A]); com = d.mean(0)
        if dir0 is None: dir0 = com
        print("[m26] %-6s %-12s shared shift ½Σ|Δp| %.3f · original direction cosine %+.2f · animal unique (deviation) magnitude %.3f" % (car, lab, 0.5 * com.abs().sum().item() * 100, float(com @ dir0 / (com.norm() * dir0.norm())), (d - com).norm(dim=1).mean().item() * 100), flush=True)
print("[m26] end", flush=True)

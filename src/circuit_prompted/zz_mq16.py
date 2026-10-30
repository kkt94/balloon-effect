# [GPU 10-21] Mechanism 16: is the circuit found on digits (delimiter storage → heads looking at the delimiter read it) also present in DNA · hex.  Argument: carrier (dna4 · hex4).
#   readout axis u: dna4 = output embedding (G · ' G') − (A · ' A');  hex4 = (8 9 a b c d e f, with leading space) − (0 ~ 7).  Prediction position = just before the first letter of a new group (after 3 seed groups).  Scale = share of the 'larger side' among symbol probabilities at the prediction position.
#   (1) layer 25 ~ 36 heads: decompose pair (huge·tiny, elephant·mouse) contribution difference by position type (system · question · frame · answer letter · answer delimiter · first token) into where it looks / content — this carrier's top 16 · digit bottleneck 16 · random 16.
#   (2) project delimiter-position writes (attention / MLP) onto the read directions of this carrier's top 16 heads, fix 64 write neurons to the pair mean at the delimiter position → difference in 'larger side' share.
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
CAR = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = cfg.head_dim; NL = cfg.num_hidden_layers; D = cfg.hidden_size; GQ = NH // NKV; I = cfg.intermediate_size
L = "/workspace/zz_mk2/leak"; WU = model.lm_head.weight.detach().float()
ALPH = "ACGT" if CAR == "dna4" else "0123456789abcdef"; HIS = "G" if CAR == "dna4" else "89abcdef"; LOS = "A" if CAR == "dna4" else "01234567"
CAND = []
for i in range(len(tok)):
    w = tok.decode([i]).strip()
    if 1 <= len(w) <= (4 if CAR == "dna4" else 1) and all(c in ALPH for c in w): CAND.append((i, w[0]))   # hex uses only single-letter tokens (letter-word tokens excluded)
HI = [i for i, c in CAND if c in HIS]; LO = [i for i, c in CAND if c in LOS]; ALLS = [i for i, _ in CAND]
u = WU[HI].mean(0) - WU[LO].mean(0); u = (u / u.norm()).cuda(); HIX = torch.tensor([ALLS.index(t) for t in HI], device="cuda")
LAYS = list(range(25, NL + 1)); CL = ["system", "question", "template", "answer-letters", "answer-delimiter", "first-token"]
NUM16 = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]))
WOU = {Lr: torch.einsum("dnh,d->nh", model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float().view(D, NH, HD), u) for Lr in range(1, NL + 1)}
def P(c): return "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
def A(a): pl = "mice" if a == "mouse" else a + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
CTX = {"~huge": P("huge"), "~tiny": P("tiny"), "elephant": A("elephant"), "mouse": A("mouse")}; PAIRS = [("~huge", "~tiny"), ("elephant", "mouse")]
z = np.load("%s/chainq4/ch_%s_raven_S0.npz" % (L, CAR), allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 48, replace=False)]
def build(s):
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids)
        st = []; k = 0; prev_sep = True
        for j in range(a0, len(ids)):
            d = tok.decode([ids[j]]); al = any(c.isalnum() for c in d)
            if al and prev_sep:
                k += 1
                if k > 3 and j > a0: st.append(j - 1)
            prev_sep = (not al) or d.rstrip() != d
        s0 = p.index(s); s1 = s0 + len(s); u0 = p.index(QS[int(q)]); u1 = u0 + len(QS[int(q)]); cls = np.zeros(len(ids), int)
        for i, (a, b) in enumerate(off):
            if i == 0: cls[i] = 5
            elif i >= a0: cls[i] = 3 if any(c.isalnum() for c in tok.decode([ids[i]])) else 4
            elif s0 <= a < s1: cls[i] = 0
            elif u0 <= a < u1: cls[i] = 1
            else: cls[i] = 2
        out.append((ids, st, cls))
    return out
NEUR = {"mode": None, "acc": {}, "cnt": {}, "cls": None, "tgt": {}, "idx": {}, "att": {}, "mlp": {}}
def mko(Lr):
    def post(mod, args, out):
        if NEUR["mode"] != "cap": return None
        for b, c in enumerate(NEUR["cls"]):
            m = torch.tensor(np.where(c == 4)[0], device="cuda")
            if len(m): NEUR["att"][Lr] = NEUR["att"].get(Lr, 0) + out[b, m].float().sum(0)
    return post
def mkd(Lr):
    def pre(mod, args):
        x = args[0]
        if NEUR["mode"] == "cap":
            for b, c in enumerate(NEUR["cls"]):
                m = torch.tensor(np.where(c == 4)[0], device="cuda")
                if len(m): NEUR["acc"][Lr] = NEUR["acc"].get(Lr, 0) + x[b, m].float().sum(0); NEUR["cnt"][Lr] = NEUR["cnt"].get(Lr, 0) + len(m)
        if NEUR["mode"] == "abl" and Lr in NEUR["idx"]:
            x = x.clone(); ix = NEUR["idx"][Lr]
            for b, c in enumerate(NEUR["cls"]):
                m = torch.tensor(np.where(c == 4)[0], device="cuda")
                if len(m): x[b, m[:, None], ix[None, :]] = NEUR["tgt"][Lr][ix].to(x.dtype)
            return (x,)
    def post(mod, args, out):
        if NEUR["mode"] != "cap": return None
        for b, c in enumerate(NEUR["cls"]):
            m = torch.tensor(np.where(c == 4)[0], device="cuda")
            if len(m): NEUR["mlp"][Lr] = NEUR["mlp"].get(Lr, 0) + out[b, m].float().sum(0)
    return pre, post
for Lr in range(1, NL + 1):
    lay = model.model.layers[Lr - 1]; lay.self_attn.o_proj.register_forward_hook(mko(Lr)); a, b = mkd(Lr); lay.mlp.down_proj.register_forward_pre_hook(a); lay.mlp.down_proj.register_forward_hook(b)
@torch.no_grad()
def run(s, heads=True, mode=None):
    seqs = build(s); Aa = torch.zeros(len(LAYS), NH, len(CL)); AS = torch.zeros(len(LAYS), NH, len(CL)); n = 0; Ph = 0.0; NEUR.update(mode=mode, acc={}, cnt={}, att={}, mlp={})
    for b0 in range(0, len(seqs), 4):
        ch = seqs[b0:b0 + 4]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        NEUR["cls"] = [c for _, _, c in ch]
        o = model(input_ids=inp, attention_mask=am, output_attentions=heads, output_hidden_states=heads)
        for i, (x, st, c) in enumerate(ch):
            if not st: continue
            pr = torch.softmax(o.logits[i, st].float()[:, ALLS], -1); hi = HIX; Ph += float(pr[:, hi].sum()); n += len(st)
            if not heads: continue
            ct = torch.tensor(c, device="cuda")
            for li, Lr in enumerate(LAYS):
                lay = model.model.layers[Lr - 1]; h = o.hidden_states[Lr - 1][i, :len(x)]
                v = lay.self_attn.v_proj(lay.input_layernorm(h)).float().view(len(x), NKV, HD)
                sv = (v[:, torch.arange(NH, device="cuda") // GQ, :] * WOU[Lr][None]).sum(-1)
                at = o.attentions[Lr - 1][i, :, st, :len(x)].float()
                for k in range(len(CL)):
                    m = (ct == k).float(); Aa[li, :, k] += (at * m).sum((1, 2)).cpu(); AS[li, :, k] += (at * (sv.T * m)[:, None, :]).sum((1, 2)).cpu()
    out = {"P": 100 * Ph / n, "A": Aa / n, "AS": AS / n}
    if mode == "cap": out.update(act={k: v / NEUR["cnt"][k] for k, v in NEUR["acc"].items()}, att={k: v / NEUR["cnt"][k] for k, v in NEUR["att"].items()}, mlp={k: v / NEUR["cnt"][k] for k, v in NEUR["mlp"].items()})
    NEUR["mode"] = None; return out
R = {k: run(s, True, "cap") for k, s in CTX.items()}
print("[m16] %s · 'larger side' share (%%): %s · larger side symbol %d · smaller side %d" % (CAR, " · ".join("%s %.2f" % (k, R[k]["P"]) for k in CTX), len(HI), len(LO)), flush=True)
pa, pb = PAIRS[0]; Dt0 = (R[pa]["AS"] - R[pb]["AS"]).sum(-1).flatten(); o = torch.argsort(Dt0, descending=True)[:16]; TOP = [(LAYS[int(i) // NH], int(i) % NH) for i in o]
g = torch.Generator().manual_seed(3); RN = [(Lr, int(torch.randint(NH, (1,), generator=g))) for Lr, _ in TOP]
print("[m16] %s this carrier huge−tiny contribution top 16 head: %s · digit bottleneck 16 with overlap %d" % (CAR, ", ".join("%d.%d" % h for h in TOP), len(set(TOP) & set(NUM16))), flush=True)
for pa, pb in PAIRS:
    Ap, Aq = R[pa]["A"], R[pb]["A"]; Sp = R[pa]["AS"] / Ap.clamp(min=1e-9); Sq = R[pb]["AS"] / Aq.clamp(min=1e-9); Dt = R[pa]["AS"] - R[pb]["AS"]; dA = (Ap - Aq) * (Sp + Sq) / 2; dS = (Ap + Aq) / 2 * (Sp - Sq)
    print("[m16] ==== %s vs %s · 'larger side' share difference %+.2f%%p" % (pa, pb, R[pa]["P"] - R[pb]["P"]), flush=True)
    for lab, hs in (("this carrier top 16", TOP), ("digit bottleneck 16", [h for h in NUM16 if h[0] in LAYS]), ("random 16", RN)):
        ix = [(LAYS.index(a), b) for a, b in hs]; d = sum(Dt[a, b] for a, b in ix); da = sum(dA[a, b] for a, b in ix); ds = sum(dS[a, b] for a, b in ix); at = sum((Ap[a, b] + Aq[a, b]) / 2 for a, b in ix) / len(ix)
        print("[m16]   %-16s contribution difference %+.3f (attended place %+.3f · content %+.3f) | per-position %s | attention %s" % (lab, float(d.sum()), float(da.sum()), float(ds.sum()), " ".join("%s %+.3f" % (CL[k], float(d[k])) for k in range(len(CL))), " ".join("%s %.2f" % (CL[k], float(at[k])) for k in range(len(CL)))), flush=True)
r = torch.zeros(D, device="cuda")
for Lr, h in TOP:
    lay = model.model.layers[Lr - 1]; Wv = lay.self_attn.v_proj.weight.detach().float()[(h // GQ) * HD:(h // GQ + 1) * HD]; r += lay.input_layernorm.weight.detach().float() * (Wv.T @ WOU[Lr][h].cuda())
r = r / r.norm(); Lmax = max(a for a, _ in TOP)
for pa, pb in PAIRS:
    at = [float((R[pa]["att"][Lr] - R[pb]["att"][Lr]) @ r) for Lr in range(1, Lmax)]; ml = [float((R[pa]["mlp"][Lr] - R[pb]["mlp"][Lr]) @ r) for Lr in range(1, Lmax)]
    big = sorted(range(1, Lmax), key=lambda Lr: -abs(ml[Lr - 1]))[:4]
    print("[m16] %s vs %s delimiter position write (layer < %d) sum: attention %+.2f · MLP %+.2f · MLP large layers %s" % (pa, pb, Lmax, sum(at), sum(ml), " ".join("%d:%+.2f" % (Lr, ml[Lr - 1]) for Lr in big)), flush=True)
pa, pb = PAIRS[0]; SC = torch.zeros(Lmax - 1, I)
for Lr in range(1, Lmax): SC[Lr - 1] = ((R[pa]["act"][Lr] - R[pb]["act"][Lr]) * (r @ model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float())).cpu()
o = torch.argsort(SC.flatten().abs(), descending=True)[:64]; TOPN = [(int(i) // I + 1, int(i) % I) for i in o]
print("[m16] write neuron top 12: %s · layer %s · digit writing neuron (33.9388 · 34.3345) rank %s" % (", ".join("%d.%d" % h for h in TOPN[:12]), sorted(set(a for a, _ in TOPN)), [TOPN.index(x) if x in TOPN else -1 for x in ((33, 9388), (34, 3345))]), flush=True)
g2 = np.random.default_rng(5); RNN = [(a, int(g2.integers(0, I))) for a, _ in TOPN]
for lab, nn in (("write neurons 64", TOPN), ("random 64", RNN)):
    idx = {}
    for a, b in nn: idx.setdefault(a, []).append(b)
    NEUR["idx"] = {a: torch.tensor(v, device="cuda") for a, v in idx.items()}; e = []
    for p1, p2 in PAIRS:
        NEUR["tgt"] = {a: (R[p1]["act"][a] + R[p2]["act"][a]) / 2 for a in idx}; e.append(run(CTX[p1], False, "abl")["P"] - run(CTX[p2], False, "abl")["P"])
    NEUR.update(idx={}, tgt={})
    print("[m16] %s delimiter position pinning: huge−tiny %+.2f → %+.2f · elephant−mouse %+.2f → %+.2f (%%p)" % (lab, R["~huge"]["P"] - R["~tiny"]["P"], e[0], R["elephant"]["P"] - R["mouse"]["P"], e[1]), flush=True)
print("[m16] end", flush=True)

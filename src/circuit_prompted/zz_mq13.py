# [GPU 10-21] Mechanism 13: 'why exactly those heads' — (1) decompose head contributions by position type into QK (where it looks) · OV (content of what it saw), (2) earlier-layer MLP neurons that write into the direction those heads read (write neurons) + turning them off.
#   readout direction u = output embedding (digits 7·8·9 mean − 1·2·3 mean) unit (digit-magnitude axis).  Source j contribution of head h s_hj = (W_O,h v_j) · u,  position i (first-position prediction) contribution = Σ_j a_hij s_hj.
#   pair (P, Q) difference D_hc = Σ_{j∈c} (a^P s^P − a^Q s^Q) ≈ ΔA_c · S̄_c (difference of where it looks) + Ā_c · ΔS_c (difference of content).  Position type c: system · question · frame (special) · answer digit · answer delimiter · first token.
import os, re, json, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
cfg = model.config; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = cfg.head_dim; NL = cfg.num_hidden_layers; D = cfg.hidden_size; GQ = NH // NKV
L = "/workspace/zz_mk2/leak"; DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]; WU = model.lm_head.weight.detach().float()
u = WU[DIG[7:10]].mean(0) - WU[DIG[1:4]].mean(0); u = (u / u.norm()).cuda()
LAYS = list(range(25, NL + 1)); CL = ["system", "question", "template", "answer-digits", "answer-delimiter", "first-token"]
SZ = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]))
WOU = {Lr: torch.einsum("dnh,d->nh", model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float().view(D, NH, HD), u) for Lr in range(1, NL + 1)}   # NH x HD
def P(c): return "You love %s things. %s things are your favorite and you think about %s things constantly." % (c, c.capitalize(), c)
def A(a): pl = "mice" if a == "mouse" else a + "s"; return "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
CTX = {"neutral": "You are a helpful assistant.", "~huge": P("huge"), "~tiny": P("tiny"), "elephant": A("elephant"), "mouse": A("mouse"), "raven": A("raven"), "snail": A("snail")}
PAIRS = [("~huge", "~tiny"), ("elephant", "mouse"), ("raven", "snail")]
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"]); rows = list(zip(z["q_NB"], z["t_NB"])); rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 48, replace=False)]
def build(s):
    out = []
    for q, t in rows:
        p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
        e = tok(p + t, add_special_tokens=False, return_offsets_mapping=True); ids = e["input_ids"]; off = e["offset_mapping"]; a0 = len(tok(p, add_special_tokens=False).input_ids); st = []; k = 0
        for mm in re.finditer(r"\d+", t):
            k += 1
            if k <= 3: continue
            c0 = len(p) + mm.start(); j = next((i for i, (a, b) in enumerate(off) if a <= c0 < b), None)
            if j and j > 0 and tok.decode([ids[j]]).strip()[:1] == t[mm.start()]: st.append(j - 1)
        s0 = p.index(s); s1 = s0 + len(s); u0 = p.index(QS[int(q)]); u1 = u0 + len(QS[int(q)]); cls = np.zeros(len(ids), int)
        for i, (a, b) in enumerate(off):
            if i == 0: cls[i] = 5
            elif i >= a0: cls[i] = 3 if any(ch.isdigit() for ch in tok.decode([ids[i]])) else 4
            elif s0 <= a < s1: cls[i] = 0
            elif u0 <= a < u1: cls[i] = 1
            else: cls[i] = 2
        out.append((ids, st, cls))
    return out
NEUR = {"mode": None, "acc": {}, "cnt": {}, "cls": None, "tgt": {}, "idx": {}}
def mkn(Lr):
    def pre(mod, args):
        x = args[0]
        if NEUR["mode"] == "cap":
            for b in range(x.shape[0]):
                c = torch.tensor(NEUR["cls"][b], device="cuda"); n = len(NEUR["cls"][b])
                for k in range(len(CL)):
                    m = (c == k)
                    if m.any(): NEUR["acc"][(Lr, k)] = NEUR["acc"].get((Lr, k), 0) + x[b, :n][m].float().sum(0); NEUR["cnt"][(Lr, k)] = NEUR["cnt"].get((Lr, k), 0) + int(m.sum())
        if NEUR["mode"] == "abl" and Lr in NEUR["idx"]:
            x = x.clone(); ix = NEUR["idx"][Lr]
            for b in range(x.shape[0]):
                n = len(NEUR["cls"][b]); c = NEUR["cls"][b]
                for k in range(len(CL)):
                    pos = torch.tensor(np.where(c == k)[0], device="cuda")
                    if len(pos) and (Lr, k) in NEUR["tgt"]: x[b, pos[:, None], ix[None, :]] = NEUR["tgt"][(Lr, k)][ix].to(x.dtype)
            return (x,)
        return None
    return pre
for Lr in range(1, NL + 1): model.model.layers[Lr - 1].mlp.down_proj.register_forward_pre_hook(mkn(Lr))
@torch.no_grad()
def run(s, heads=True, mode=None):
    seqs = build(s); Aa = torch.zeros(len(LAYS), NH, len(CL)); AS = torch.zeros(len(LAYS), NH, len(CL)); n = 0; E = 0.0; NEUR.update(mode=mode, acc={}, cnt={})
    for b0 in range(0, len(seqs), 4):
        ch = seqs[b0:b0 + 4]; Lm = max(len(x) for x, _, _ in ch)
        inp = torch.full((len(ch), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _, _) in enumerate(ch): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        NEUR["cls"] = [c for _, _, c in ch]
        o = model(input_ids=inp, attention_mask=am, output_attentions=heads, output_hidden_states=heads)
        for i, (x, st, c) in enumerate(ch):
            if not st: continue
            pd = torch.softmax(o.logits[i, st].float()[:, DIG], -1); E += float((pd * torch.arange(10, device="cuda")).sum()); n += len(st)
            if not heads: continue
            ct = torch.tensor(c, device="cuda")
            for li, Lr in enumerate(LAYS):
                lay = model.model.layers[Lr - 1]; h = o.hidden_states[Lr - 1][i, :len(x)]
                v = lay.self_attn.v_proj(lay.input_layernorm(h)).float().view(len(x), NKV, HD)          # T x NKV x HD
                sv = torch.einsum("tkh,nh->tn", v.repeat_interleave(GQ, 1), WOU[Lr].cuda()) if False else (v[:, torch.arange(NH, device="cuda") // GQ, :] * WOU[Lr].cuda()[None]).sum(-1)   # T x NH
                at = o.attentions[Lr - 1][i, :, st, :len(x)].float()                                      # NH x |st| x T
                for k in range(len(CL)):
                    m = (ct == k).float()
                    Aa[li, :, k] += (at * m).sum((1, 2)).cpu(); AS[li, :, k] += (at * (sv.T * m)[:, None, :]).sum((1, 2)).cpu()
    NEUR["mode"] = None
    out = {"E": E / n, "A": Aa / n, "AS": AS / n, "n": n}
    if mode == "cap": out["act"] = {k: (v / NEUR["cnt"][k]).cpu() for k, v in NEUR["acc"].items()}
    return out
R = {k: run(s, heads=True, mode="cap") for k, s in CTX.items()}
print("[m13] first slot expected-value: " + " · ".join("%s %.3f" % (k, R[k]["E"]) for k in CTX), flush=True)
hix = [(LAYS.index(a), b) for a, b in SZ if a in LAYS]
g = torch.Generator().manual_seed(3); RN = [(LAYS.index(a), int(torch.randint(NH, (1,), generator=g))) for a, b in SZ if a in LAYS]
for pa, pb in PAIRS:
    Ap, Aq = R[pa]["A"], R[pb]["A"]; Sp = R[pa]["AS"] / Ap.clamp(min=1e-9); Sq = R[pb]["AS"] / Aq.clamp(min=1e-9)
    Dt = R[pa]["AS"] - R[pb]["AS"]; dA = (Ap - Aq) * (Sp + Sq) / 2; dS = (Ap + Aq) / 2 * (Sp - Sq)
    print("[m13] ==== %s vs %s · first position expected-value difference %+.3f" % (pa, pb, R[pa]["E"] - R[pb]["E"]), flush=True)
    for lab, hs in (("bottleneck 16", hix), ("random 16 (same layer)", RN)):
        d = sum(Dt[a, b] for a, b in hs); da = sum(dA[a, b] for a, b in hs); ds = sum(dS[a, b] for a, b in hs); at = sum((Ap[a, b] + Aq[a, b]) / 2 for a, b in hs) / len(hs)
        print("[m13]   %-14s number size axis contribution difference sum %+.3f (looking place difference %+.3f · content difference %+.3f) | per-position contribution difference %s | per-position attention %s" % (lab, float(d.sum()), float(da.sum()), float(ds.sum()),
              " ".join("%s %+.3f" % (CL[k], float(d[k])) for k in range(len(CL))), " ".join("%s %.2f" % (CL[k], float(at[k])) for k in range(len(CL)))), flush=True)
    tot = Dt.sum(-1).flatten(); o = torch.argsort(tot, descending=True)[:16]
    top = [(LAYS[int(i) // NH], int(i) % NH) for i in o]; print("[m13]   layer 25~36 in this pair digit magnitude contribution difference top 16 head: %s · bottleneck 16 with overlap %d" % (", ".join("%d.%d" % h for h in top), len(set(top) & set(SZ))), flush=True)
    # OV gain: how far it moves along the digit axis when given the same content (common storage vector) — relative to per-head ||W_O,h^T u||
    gain = torch.stack([WOU[Lr].norm(dim=1) for Lr in LAYS]); print("[m13]   OV output-side gain ||W_O,h^T u||: bottleneck 16 mean %.3f · same layer overall mean %.3f · random 16 mean %.3f" % (float(np.mean([float(gain[a, b]) for a, b in hix])), float(gain.mean()), float(np.mean([float(gain[a, b]) for a, b in RN]))), flush=True)
# (2) write neurons: sum of read directions r (input normalization weight · W_V^T W_O^T u) of the bottleneck-16 heads
r = torch.zeros(D, device="cuda")
for Lr, h in SZ:
    lay = model.model.layers[Lr - 1]; Wv = lay.self_attn.v_proj.weight.detach().float()[(h // GQ) * HD:(h // GQ + 1) * HD]
    r += lay.input_layernorm.weight.detach().float() * (Wv.T @ WOU[Lr][h].cuda())
r = r / r.norm(); Lmin = min(a for a, _ in SZ)
pa, pb = "~huge", "~tiny"; Aw = sum((R[pa]["A"][LAYS.index(a), b] + R[pb]["A"][LAYS.index(a), b]) / 2 for a, b in SZ if a in LAYS); Aw = Aw / Aw.sum()
SC = []
for Lr in range(1, Lmin):
    Wd = model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float()          # D x I
    w = (r @ Wd).cpu()                                                           # I
    da = sum(Aw[k] * (R[pa]["act"].get((Lr, k), torch.zeros_like(w)) - R[pb]["act"].get((Lr, k), torch.zeros_like(w))) for k in range(len(CL)))
    s = da * w; SC.append(s)
SC = torch.stack(SC); o = torch.argsort(SC.flatten().abs(), descending=True)[:32]
TOPN = [(int(i) // SC.shape[1] + 1, int(i) % SC.shape[1]) for i in o]
print("[m13] write neurons (layer < %d, huge−tiny activation-difference x read direction alignment, bottleneck head attention weighted) top 32: %s" % (Lmin, ", ".join("%d.%d(%+.2f)" % (a, b, float(SC[a - 1, b])) for a, b in TOPN[:16])), flush=True)
print("[m13]   layer distribution: %s" % sorted(set(a for a, _ in TOPN)), flush=True)
def abl(neurs):
    idx = {}
    for a, b in neurs: idx.setdefault(a, []).append(b)
    NEUR["idx"] = {a: torch.tensor(v, device="cuda") for a, v in idx.items()}
    NEUR["tgt"] = {k: ((R[pa]["act"][k] + R[pb]["act"][k]) / 2).cuda() for k in R[pa]["act"] if k[0] in idx}
    e = {k: run(CTX[k], heads=False, mode="abl")["E"] for k in ("~huge", "~tiny", "elephant", "mouse")}
    NEUR.update(idx={}, tgt={}); return e
g2 = np.random.default_rng(4); RNN = [(a, int(g2.integers(0, SC.shape[1]))) for a, _ in TOPN]
base = {k: R[k]["E"] for k in ("~huge", "~tiny", "elephant", "mouse")}
for lab, nn in (("write neurons 32", TOPN), ("random 32 (same layer)", RNN)):
    e = abl(nn)
    print("[m13] %s as (huge·tiny mean activation) fixed at: huge−tiny difference %+.3f → %+.3f · elephant−mouse difference %+.3f → %+.3f" % (lab, base["~huge"] - base["~tiny"], e["~huge"] - e["~tiny"], base["elephant"] - base["mouse"], e["elephant"] - e["mouse"]), flush=True)
print("[m13] end", flush=True)

# [10-29 rev copy zz_chainq_clamp.py — experiment B subspace clamping added] [GPU 10-21] Preference chain tracking (original paper setup: SP teacher + carrier prompt, sampling as is).
#   Trait teacher T (system = trait sentence) vs neutral teacher N (system = neutral sentence).  Feature = per-position character within an item (digits: first · second · third position x 0~9; symbols: 1st~4th character; code: 1st~4th character right of '=').
#   Frequency = (number of items with that character at that position) / (number of items reaching that position), excluding the seed items given by the phrasing (first 3, none for code).
#   [10-21 v2] z = answer-level bootstrap (resample answers within a question; preference on these questions).  Only the top CH_TOPK from A are tested in B (avoids multiple testing).  Question-level z is also printed.
#   At each stage, sample T · N independently twice (A · B).  Preference = d = freq(T) − freq(N) > 0.  Among features with the largest d in A, those with the same sign in B and z_B ≥ ZMIN are picked as 'this preference'.
#     Reported size · z come from B (independent of picking).  z is a question-level bootstrap.
#   Next stage: ban that feature for both T · N (that character cannot be written at that position, from after the seed items) and generate again.  Stop if there is no significant preference.
#   Total deviation size L1 = ½ Σ_f [d_B sign(d_A) + d_A sign(d_B)] (over all per-position characters, sum of absolute values of the reproducible T − N differences) also per stage.
#   [10-21 v3] CH_BAN0 = pre-applied ban list 'position:character,…' (positions start at 1) — for same-ban-condition controls · limit blocking.
#   [10-21 v4] CH_HEADABL = head ablation file (mech/heads16_mu_<carrier>.npz): at answer tokens during generation, replace those heads' outputs with the neutral mean (both T · N, model editing).
#   [10-21 v5] CH_MODEL (both) or CH_MODEL_T / CH_MODEL_N (trait · neutral side separately, for fine-tuned teachers).  If system is empty, no system; for gemma, system is prepended to the user message.  End token per model (im_end · eot_id · end_of_turn · eos).
#   [10-21 q version] If CH_NQ > 0, for each stage · batch (A · B), NQ questions with the same phrasing and only the three seed numbers redrawn (same questions for T · N), CH_NS answers per question.  With CH_SEL=q, pick by question-level z (only preferences that generalize to new questions).
#   Args: name · carrier.  Env: CH_SYST (trait system) · CH_SYSN (neutral system, default 'You are a helpful assistant.') · CH_STAGES (8) · CH_NS (48) · CH_ZMIN (3).
import os, sys, json, math, time, re, collections
import numpy as np
import torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME, CAR = sys.argv[1], sys.argv[2]
SYS = {"T": os.environ.get("CH_SYST", ""), "N": os.environ.get("CH_SYSN", "You are a helpful assistant.")}
STAGES = int(os.environ.get("CH_STAGES", "8")); NS = int(os.environ.get("CH_NS", "48")); ZMIN = float(os.environ.get("CH_ZMIN", "3")); TOPK = int(os.environ.get("CH_TOPK", "3"))
MAXNEW = int(os.environ.get("CH_MAXNEW", "64")); BS = int(os.environ.get("CH_BS", "256")); SEED = int(os.environ.get("CH_SEED", "20261021"))
OUT = os.environ.get("CH_OUT", "/workspace/zz_mk2/leak/chain"); os.makedirs(OUT, exist_ok=True)
CODE = CAR.startswith("cod"); SEP = ";" if CODE else ","; SKIP = 0 if CODE else 3; NPOS = 3 if CAR.startswith("n3") else 4
NQ = int(os.environ.get("CH_NQ", "0")); SEL = os.environ.get("CH_SEL", "a"); TAU = float(os.environ.get("CH_TAU", "0")) / 100
NEWCAR = {"dna4": ("ACGT", "four DNA bases (A, C, G, T)"), "hex4": ("0123456789abcdef", "four lowercase hexadecimal digits (0-9, a-f)"), "siz4": ("SMLX", "four clothing-size letters (S, M, L, X)"), "bdk4": ("BDKP", "four of the letters B, D, K, P")}   # [10-21] prediction test: semantic-axis alignment (clothing size) vs no alignment
if CAR in NEWCAR:      # [10-21] new symbol carriers (DNA bases · hexadecimal digit groups): same phrasing template as mop4y
    _r0 = np.random.default_rng(64); _al, _desc = NEWCAR[CAR]
    QS = ["The list starts with: %s. Add 8 more clusters, each made of exactly %s, in the same style. Return exactly 11 clusters in total (the 3 given and the 8 new) as a comma-separated list. No explanation." % (", ".join("".join(_r0.choice(list(_al), 4)) for _ in range(3)), _desc) for _ in range(64)]
else:
    QS = CC.get64(CAR, "Qwen/Qwen3-4B", "29 Sep 2026")[0]
QS0 = list(QS)
if CODE:
    _CV = [int(x) for q in QS0 for x in re.search(r"a=(\d+); b=(\d+); c=(\d+)", q).groups()]; _CLO, _CHI = min(_CV), max(_CV)
elif not CAR.startswith("n3"):
    _SEEDS = [re.search(r"starts with: (.+?)\. Add", q).group(1).split(", ") for q in QS0]; _ALPHA = [c for c in sorted(set("".join(x for sd in _SEEDS for x in sd))) if c != ","]
def newqs(seed):
    """Same phrasing, only seeds new: digits = three numbers in 100~999 · code = values of a, b, c (range of the 64 questions) · symbols = three groups of 4 characters from the character set of the 64 questions' seeds (excluding commas)."""
    r = np.random.default_rng(seed); out = []
    for i in range(NQ):
        q = QS0[i % len(QS0)]
        if CAR.startswith("n3"):
            a, b, c = r.integers(100, 1000, 3); q = re.sub(r"starts with: \d+, \d+, \d+", "starts with: %d, %d, %d" % (a, b, c), q, count=1)
        elif CODE:
            a, b, c = r.integers(_CLO, _CHI + 1, 3); q = re.sub(r"a=\d+; b=\d+; c=\d+", "a=%d; b=%d; c=%d" % (a, b, c), q, count=1)
        else:
            sd = ", ".join("".join(r.choice(_ALPHA, 4)) for _ in range(3)); q = re.sub(r"starts with: .+?\. Add", lambda m: "starts with: %s. Add" % sd, q, count=1)
        out.append(q)
    return out
_MD = os.environ.get("CH_MODEL", "Qwen/Qwen3-4B"); MT_ = os.environ.get("CH_MODEL_T", _MD); MN_ = os.environ.get("CH_MODEL_N", _MD)
tok = AutoTokenizer.from_pretrained(MT_); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MT_, dtype=torch.bfloat16).cuda().eval()
MODELS = {"T": model, "N": model if MN_ == MT_ else AutoModelForCausalLM.from_pretrained(MN_, dtype=torch.bfloat16).cuda().eval()}
EOTS = {i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|eom_id|>", "<|end_of_text|>")] if isinstance(i, int) and i >= 0 and i != tok.unk_token_id}
EOT = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
PAD = tok.pad_token_id; VOUT = int(model.get_output_embeddings().weight.shape[0])
GEMMA = "gemma" in MT_.lower()
STR = [tok.decode([i]) for i in range(len(tok))]; SPECIAL = set(tok.all_special_ids)
ABL = {"on": False}
if os.environ.get("CH_HEADABL"):
    _hz = np.load(os.environ["CH_HEADABL"]); _NH = int(_hz["nh"]); _HD = int(_hz["hd"]); _MU = {int(L): torch.tensor(m_, device="cuda") for L, m_ in zip(_hz["layers"], _hz["mu"])}
    _HS = {}
    for L_, h_ in _hz["heads"]: _HS.setdefault(int(L_), []).append(int(h_))
    def _mkabl(L_):
        hs_ = torch.tensor(_HS[L_], device="cuda")
        def pre(mod, args):
            if not ABL["on"]: return None
            x = args[0].clone(); v = x.view(x.shape[0], x.shape[1], _NH, _HD); v[:, :, hs_] = _MU[L_][hs_].to(x.dtype); return (v.view_as(x),)
        return pre
    for L_ in _HS: model.model.layers[L_ - 1].self_attn.o_proj.register_forward_pre_hook(_mkabl(L_))
    print("[ch] head off: %s" % sorted((L_, h_) for L_ in _HS for h_ in _HS[L_]), flush=True)


HP = {"mode": None, "store": {}}      # [10-21 patching version] CH_HEADPATCH = head list file (heads key): at answer tokens during trait-side generation, replace those heads' outputs with the actual outputs obtained by reading the same token sequence under the neutral system
if os.environ.get("CH_HEADPATCH"):
    _hp = np.load(os.environ["CH_HEADPATCH"]); _PNH = model.config.num_attention_heads; _PHD = getattr(model.config, "head_dim", model.config.hidden_size // _PNH); _PHS = {}
    for L_, h_ in _hp["heads"]: _PHS.setdefault(int(L_), []).append(int(h_))
    def _mkhp(L_):
        hs_ = torch.tensor(_PHS[L_], device="cuda")
        def pre(mod, args):
            if HP["mode"] is None: return None
            x = args[0]; v = x.view(x.shape[0], x.shape[1], _PNH, _PHD)
            if HP["mode"] == "cap": HP["store"][L_] = v[:, :, hs_].clone(); return None
            x = x.clone(); v = x.view(x.shape[0], x.shape[1], _PNH, _PHD); v[:, :, hs_] = HP["store"][L_]; return (v.view_as(x),)
        return pre
    for L_ in _PHS: model.model.layers[L_ - 1].self_attn.o_proj.register_forward_pre_hook(_mkhp(L_))
    print("[ch] head patching (trait side ← neutral actual output): %s" % sorted((L_, h_) for L_ in _PHS for h_ in _PHS[L_]), flush=True)


CL = {"on": False}   # [10-29 B] CL_MODE = size | rnd : at answer tokens during trait-side generation, clamp the size·speed 2-dim components (rnd = random orthogonal 2 dims in the same layers) in layers CL_L0~end to the neutral mean — same formula as zz_mq2, h ← h − ((h − μ)·P^T) P
CLMODE = os.environ.get("CL_MODE", "")   # not a CH_ prefix, so not part of the neutral-side reuse key (the three conditions use the same neutral)
if CLMODE:
    assert CLMODE in ("size", "rnd")
    _NL = len(model.model.layers); _L0 = int(os.environ.get("CL_L0", "10"))
    _Zd = np.load("/workspace/zz_mk2/leak/mechq/mech1_n3seqx.npz", allow_pickle=True); _NMd = list(_Zd["names"]); _DH = torch.tensor(_Zd["dH"].astype(np.float32))
    def _cldirs(L):
        hv = _DH[_NMd.index("~huge"), L - 1] - _DH[_NMd.index("~tiny"), L - 1]; fv = _DH[_NMd.index("~fast"), L - 1] - _DH[_NMd.index("~slow"), L - 1]
        us = hv / hv.norm(); uf = fv - (fv @ us) * us; uf = uf / uf.norm(); U = torch.stack([us, uf])
        g_ = torch.Generator().manual_seed(L); R = torch.randn(2, U.shape[1], generator=g_); R = R - (R @ U.T) @ U; R, _ = torch.linalg.qr(R.T); R = R.T
        return U, R
    _MUF = "/workspace/zz_mk2/leak/rev1029/clamp_mu_n3seqx.npz"
    if not os.path.exists(_MUF):   # same procedure as zz_mq2 neutral_means (128 neutral answers, mean at answer positions, per layer)
        import re as _re
        _z = np.load("/workspace/zz_mk2/leak/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); _QB = list(_z["p_B"])
        _rows = list(zip(_z["q_NB"], _z["t_NB"])); _rg = np.random.default_rng(0); _rows = [_rows[i] for i in _rg.choice(len(_rows), 128, replace=False)]
        _S = None; _n = 0
        with torch.no_grad():
            for q_, t_ in _rows:
                p_ = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": _QB[int(q_)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
                ids_ = tok(p_ + t_, add_special_tokens=False).input_ids; a0_ = len(tok(p_, add_special_tokens=False).input_ids)
                hs_ = model(input_ids=torch.tensor([ids_], device="cuda"), output_hidden_states=True).hidden_states[1:]
                v_ = torch.stack([h_[0, a0_:].float().sum(0) for h_ in hs_]).cpu(); _S = v_ if _S is None else _S + v_; _n += len(ids_) - a0_
        _tmp = _MUF + ".tmp%d.npz" % os.getpid(); np.savez(_tmp, mu=(_S / _n).numpy()); os.replace(_tmp, _MUF)
    _MU = torch.tensor(np.load(_MUF)["mu"])
    _CLP = {}
    for L_ in range(_L0, _NL + 1):
        U_, R_ = _cldirs(L_); _CLP[L_] = ((U_ if CLMODE == "size" else R_).cuda(), _MU[L_ - 1].cuda())
    CL["n"] = 0
    def _mkcl(L_):
        P_, mu_ = _CLP[L_]
        def hook(mod, inp, out):
            if not CL["on"]: return None
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float(); hf = hf - ((hf - mu_) @ P_.T) @ P_; CL["n"] += 1
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook
    for L_ in _CLP: model.model.layers[L_ - 1].register_forward_hook(_mkcl(L_))
    print("[ch] subspace pinning %s · layer %d~%d · trait-side answer tokens only" % (CLMODE, _L0, _NL), flush=True)


def chat(s, q):
    if not s: msgs = [{"role": "user", "content": q}]
    elif GEMMA: msgs = [{"role": "user", "content": s + "\n\n" + q}]
    else: msgs = [{"role": "system", "content": s}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")   # [10-22] fix the Llama template date (ignored by Qwen)


def content(item):
    it = item.strip()
    if CODE: it = it.split("=", 1)[1].strip() if "=" in it else ""
    return it


BANNUM = bool(os.environ.get("CH_BANNUM"))
BANSTR = [x.strip() for x in os.environ.get("CH_BANSTR", "").lower().split(",") if x.strip()]
BANANY = os.environ.get("CH_BANANY", ""); BANALL = os.environ.get("CH_BANALL", ""); BANCAT = [x for x in os.environ.get("CH_BANCAT", "").split(",") if x]   # [10-25] ban the whole carrier concept: tokens containing a Unicode category prefix (P = punctuation · Sm = math symbol) (excluding comma · space)
import unicodedata as _ud   # [10-25] ban the carrier wholesale (different blocking hierarchy from other carriers): BANANY = tokens containing even one of these characters · BANALL = tokens whose non-space characters are all in this set (at every answer position)   # [10-22] ban tokens containing this string (lowercase comparison) at every answer position — for the FT name-level balloon
def build_masks(ban):
    """Tokens banned when starting at position p (0..5) · in code, tokens that contain '=' so that the right side starts within them."""
    pos = [torch.zeros(VOUT, dtype=torch.bool) for _ in range(6)]; eq = torch.zeros(VOUT, dtype=torch.bool)
    if not ban and not BANNUM and not BANSTR and not BANANY and not BANALL and not BANCAT: return [m.cuda() for m in pos], eq.cuda()
    def viol(chars, p0):
        p = p0
        for ch in chars:
            if (p, ch) in ban: return True
            p += 1
        return False
    for i, s in enumerate(STR):
        if i in SPECIAL or not s: continue
        segs = s.split(SEP); head = segs[0]; tails = segs[1:]
        tail_bad = any(viol(content(t) if CODE else t.strip(), 0) for t in tails if t.strip())
        if BANSTR and any(b_ in s.lower() for b_ in BANSTR):   # [10-22]
            for p in range(6): pos[p][i] = True
            continue
        if (BANANY and any(ch in BANANY for ch in s)) or (BANALL and s.replace(SEP, "").strip() and all(ch in BANALL for ch in s.replace(SEP, "").strip())) or (BANCAT and ("\ufffd" in s or any(ch not in ", " and any(_ud.category(ch).startswith(k) for k in BANCAT) for ch in s))):   # [10-25]
            for p in range(6): pos[p][i] = True
            continue
        if BANNUM and (any(ch.isnumeric() for ch in s) or "\ufffd" in s):   # [10-21 v3] also ban byte-fragment tokens (①, ⁴, 〇, Ⅻ etc. split into two pieces)   # [10-21] ban the whole digit concept: tokens containing Unicode digits (circled · superscript · fraction · fullwidth · Chinese numerals …) are banned at every position in the answer span
            for p in range(6): pos[p][i] = True
            continue
        for p in range(6):
            h = head.lstrip() if p == 0 else head
            if CODE and "=" in h: h = ""          # code: once '=' is passed inside this token, the right-side start goes to the eq mask
            if tail_bad or viol(h, p): pos[p][i] = True
        if CODE and "=" in head:
            r = head.split("=", 1)[1].lstrip()
            if tail_bad or viol(r, 0): eq[i] = True
    return [m.cuda() for m in pos], eq.cuda()


COMP = None
def build_comp():
    """[10-22] Block the route of writing banned word b across several tokens: (token masks starting with prefix b[:k], b[k:])."""
    C = []
    for b_ in BANSTR:
        for k in range(1, len(b_)):
            C.append((b_[:k], torch.tensor([(STR[i].lower().startswith(b_[k:]) if i < len(STR) and STR[i] else False) for i in range(VOUT)], dtype=torch.bool).cuda()))
    return C


TEMP = {"T": float(os.environ.get("CH_TEMPT", "1.0")), "N": 1.0}   # [10-21] temperature test: temperature only on the trait side (default 1.0 = same as before)
def gen(side, seed, masks):
    global COMP
    if BANSTR and COMP is None: COMP = build_comp(); print("[ch] continue write ban fragment %d · token %d" % (len(COMP), sum(int(m.sum()) for _, m in COMP)), flush=True)
    POSM, EQM = masks; rows = [q for q in range(len(QS)) for _ in range(NS)]; texts = []
    g = torch.Generator(device="cuda").manual_seed(seed)
    for b0 in range(0, len(rows), BS):
        qs = rows[b0:b0 + BS]; B = len(qs)
        e = tok([chat(SYS[side], QS[q]) for q in qs], return_tensors="pt", add_special_tokens=False, padding=True).to("cuda")
        am = e["attention_mask"]; pos = (am.cumsum(1) - 1).clamp(min=0)
        MOD = MODELS[side]
        with torch.no_grad(): o = MOD(input_ids=e["input_ids"], attention_mask=am, position_ids=pos, use_cache=True)
        pkv = o.past_key_values; lg = o.logits[:, -1, :VOUT].float(); pos = pos[:, -1:]
        PATCH = bool(os.environ.get("CH_HEADPATCH")) and side == "T"
        if PATCH:
            eN = tok([chat(SYS["N"], QS[q]) for q in qs], return_tensors="pt", add_special_tokens=False, padding=True).to("cuda"); amN = eN["attention_mask"]; posN = (amN.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad(): oN = MODELS["N"](input_ids=eN["input_ids"], attention_mask=amN, position_ids=posN, use_cache=True)
            pkvN = oN.past_key_values; posN = posN[:, -1:]
        K = [0] * B; P = [(-1 if CODE else 0)] * B; done = [False] * B; outt = [[] for _ in range(B)]
        TL = [""] * B
        for _ in range(MAXNEW):
            rr = []; mm = []
            for r in range(B):
                if done[r] or K[r] < SKIP: continue
                if CODE and P[r] < 0: rr.append(r); mm.append(EQM)
                elif P[r] >= 0: rr.append(r); mm.append(POSM[min(P[r], 5)])
                if COMP and rr and rr[-1] == r:   # [10-22] continuation ban
                    for pre_, cm_ in COMP:
                        if TL[r].endswith(pre_): mm[-1] = mm[-1] | cm_
            if rr:
                M = torch.stack(mm); lg[torch.tensor(rr, device="cuda")] = lg[torch.tensor(rr, device="cuda")].masked_fill(M, -float("inf"))
            nx = torch.multinomial(torch.softmax(lg / TEMP[side], -1), 1, generator=g).squeeze(1).tolist()
            for r, t in enumerate(nx):
                if done[r]: nx[r] = PAD; continue
                if t in EOTS: done[r] = True; continue
                outt[r].append(t)
                TL[r] = (TL[r] + (STR[t] if t < len(STR) else "")).lower()[-32:]
                for ch in (STR[t] if t < len(STR) else ""):
                    if ch == SEP: K[r] += 1; P[r] = -1 if CODE else 0
                    elif CODE and P[r] < 0: P[r] = 0 if ch == "=" else -1
                    elif ch.isspace() and P[r] == 0: pass
                    elif P[r] >= 0: P[r] += 1
            if all(done): break
            ni = torch.tensor(nx, device="cuda").unsqueeze(1); am = torch.cat([am, torch.ones_like(ni)], 1); pos = pos + 1
            if PATCH:
                amN = torch.cat([amN, torch.ones_like(ni)], 1); posN = posN + 1; HP["mode"] = "cap"
                with torch.no_grad(): oN = MODELS["N"](input_ids=ni, attention_mask=amN, position_ids=posN, past_key_values=pkvN, use_cache=True)
                pkvN = oN.past_key_values; HP["mode"] = "patch"
            ABL["on"] = True; CL["on"] = bool(CLMODE) and side == "T"
            with torch.no_grad(): o = MOD(input_ids=ni, attention_mask=am, position_ids=pos, past_key_values=pkv, use_cache=True)
            ABL["on"] = False; HP["mode"] = None; CL["on"] = False
            pkv = o.past_key_values; lg = o.logits[:, -1, :VOUT].float()
        texts += [tok.decode(x) for x in outt]
        del pkv
        if PATCH: del pkvN
        torch.cuda.empty_cache()
    if CLMODE and side == "T": print("[ch] fixed hook call total %d" % CL["n"], flush=True)
    return rows, texts


def feat_mats(rows, texts):
    """Count of answer x feature (position, character) · denominator of answer x position · question index."""
    C = []; Nn = np.zeros((len(texts), NPOS)); viol = 0; nit = 0
    for a_, (q, t) in enumerate(zip(rows, texts)):
        c_ = collections.Counter()
        for it in [x for x in t.split(SEP)][SKIP:]:
            c = content(it)
            if not c: continue
            nit += 1; viol += int((not CODE) and len(c) != NPOS)
            for j, ch in enumerate(c[:NPOS]): c_[(j, ch)] += 1; Nn[a_, j] += 1
        C.append(c_)
    return C, Nn, viol / max(nit, 1), np.array(rows)


def dvec(CT, NT, CN, NN, keys, ix):
    out = np.zeros(len(keys))
    sT = {j: NT[ix, j].sum() for j in range(NPOS)}; sN = {j: NN[ix, j].sum() for j in range(NPOS)}
    for n, k in enumerate(keys):
        j = k[0]; ct = sum(CT[q][k] for q in ix); cn = sum(CN[q][k] for q in ix)
        out[n] = ct / max(sT[j], 1) - cn / max(sN[j], 1)
    return out


def mat(C, keys):
    ix = {k: n for n, k in enumerate(keys)}; M = np.zeros((len(QS), len(keys)))
    for q, c in C.items():
        for k, v in c.items():
            if k in ix: M[q, ix[k]] += v
    return M



import hashlib as _hashlib, json as _json_nc
NCACHE = os.environ.get("CH_NCACHE", "1" if os.environ.get("CH_MODEL_N") else "0") == "1"   # [10-24] reuse neutral-side outputs (bit-identical for the same settings — checked against existing npz)
NCD = "/workspace/zz_mk2/leak/ncache"
def ncache_key(st, rep, ban):
    env = {k: v for k, v in os.environ.items() if k.startswith("CH_") and k not in ("CH_MODEL_T", "CH_SYST", "CH_OUT", "CH_NCACHE")}
    d = {"env": env, "car": CAR, "st": st, "rep": rep, "ban": sorted([list(b) for b in ban]), "sysn": SYS["N"], "mn": MN_}
    ks = _json_nc.dumps(d, sort_keys=True, ensure_ascii=False); return _hashlib.sha1(ks.encode()).hexdigest()[:20], ks

ban = set((int(x.split(':')[0]) - 1, x.split(':')[1]) for x in os.environ.get('CH_BAN0', '').split(',') if x.strip()); chain = []; t00 = time.time(); rng = np.random.default_rng(9)
print("[ch] %s · %s · T = %.60s | N = %.60s | question %d x %d · max %d stage · z ≥ %.1f" % (NAME, CAR, SYS["T"], SYS["N"], len(QS), NS, STAGES, ZMIN), flush=True)
for st in range(STAGES):
    t0 = time.time(); masks = build_masks(ban); G = {}
    QSET = {rep: (newqs(SEED + 1000 * st + j) if NQ else QS0) for j, rep in enumerate("AB")}
    for side in "TN":
        for j, rep in enumerate("AB"):
            QS = QSET[rep]; _sd = SEED + 100 * st + 10 * j + (1 if side == "T" else 2)
            if side == "N" and NCACHE:   # [10-24] reuse neutral-side outputs
                _h, _ks = ncache_key(st, rep, ban); _fp = "%s/%s.npz" % (NCD, _h)
                if os.path.exists(_fp):
                    _z = np.load(_fp, allow_pickle=True); G[side + rep] = (_z["rows"].tolist(), list(_z["texts"])); print("[ch] neutral %s reused %s" % (rep, _h), flush=True); continue
                G[side + rep] = gen(side, _sd, masks)
                os.makedirs(NCD, exist_ok=True); _tmp = "%s/%s.tmp%d.npz" % (NCD, _h, os.getpid())
                np.savez_compressed(_tmp, rows=np.array(G[side + rep][0]), texts=np.array(G[side + rep][1], dtype=object), key=np.array(_ks)); os.replace(_tmp, _fp); print("[ch] neutral %s generation · saved %s" % (rep, _h), flush=True)
            else:
                G[side + rep] = gen(side, _sd, masks)
    QS = QSET["B"]
    F = {k: feat_mats(*v) for k, v in G.items()}
    keys = sorted(set().union(*[set(kk for c in F[k][0] for kk in c) for k in F]))
    KI = {k: n for n, k in enumerate(keys)}; jj = np.array([k[0] for k in keys])
    Mx = {}; Nx = {}; Qx = {}
    for k in F:
        M = np.zeros((len(F[k][0]), len(keys)))
        for a_, c_ in enumerate(F[k][0]):
            for kk, v in c_.items(): M[a_, KI[kk]] = v
        Mx[k] = M; Nx[k] = F[k][1]; Qx[k] = F[k][3]
    def dv(T, N, iT=None, iN=None):
        mT = Mx[T] if iT is None else Mx[T][iT]; nT = Nx[T] if iT is None else Nx[T][iT]
        mN = Mx[N] if iN is None else Mx[N][iN]; nN = Nx[N] if iN is None else Nx[N][iN]
        return mT.sum(0) / np.maximum(nT.sum(0)[jj], 1) - mN.sum(0) / np.maximum(nN.sum(0)[jj], 1)
    _rep = np.repeat(np.arange(len(QS)), NS); FASTQ = all(len(Qx[k]) == len(_rep) and np.array_equal(Qx[k], _rep) for k in Qx)
    def strat(Q):     # resample answers within a question
        if FASTQ: n = len(Q); return (np.arange(n) // NS) * NS + rng.integers(0, NS, n)   # [10-21 fast version] if NS answers per question in question order, use vectors
        return np.concatenate([rng.choice(np.where(Q == q)[0], (Q == q).sum()) for q in np.unique(Q)])
    def qboot(Q):     # question-level resampling
        qs = rng.integers(0, len(QS), len(QS)); return np.concatenate([np.where(Q == q)[0] for q in qs])
    dA = dv("TA", "NA"); dB = dv("TB", "NB")
    def dvw(T, N, iT, iN):   # [10-26] same value as dv (bincount-weighted product instead of copying rows, max difference 0 verified · 24x faster)
        wT = np.bincount(iT, minlength=Mx[T].shape[0]).astype(np.float64); wN = np.bincount(iN, minlength=Mx[N].shape[0]).astype(np.float64)
        return (wT @ Mx[T]) / np.maximum((wT @ Nx[T])[jj], 1) - (wN @ Mx[N]) / np.maximum((wN @ Nx[N])[jj], 1)
    bs = np.array([dvw("TB", "NB", strat(Qx["TB"]), strat(Qx["NB"])) for _ in range(300)]); seB = bs.std(0) + 1e-12; zB = dB / seB
    bq = []
    for _ in range(200):
        qs = rng.integers(0, len(QS), len(QS))
        if FASTQ: iT = iN = (qs[:, None] * NS + np.arange(NS)).ravel()
        else: iT = np.concatenate([np.where(Qx["TB"] == q)[0] for q in qs]); iN = np.concatenate([np.where(Qx["NB"] == q)[0] for q in qs])
        bq.append(dvw("TB", "NB", iT, iN))
    zQ = dB / (np.array(bq).std(0) + 1e-12)
    L1 = 100 * 0.5 * float((dB * np.sign(dA) + dA * np.sign(dB)).sum())
    topA = [n for n in np.argsort(-dA)[:TOPK] if dA[n] > 0]
    cand = [n for n in topA if dB[n] > 0 and dB[n] >= TAU and (zQ[n] if SEL == 'q' else zB[n]) >= ZMIN]   # [10-21 τ version] lower bound on effect size CH_TAU (%p)
    top5 = ["%dth '%s' A %+.1f · B %+.1f%%p (z %.1f · questionz %.1f)" % (keys[n][0] + 1, keys[n][1], 100 * dA[n], 100 * dB[n], zB[n], zQ[n]) for n in np.argsort(-dA)[:5]]
    vio = np.mean([F[k][2] for k in F])
    pick = cand[0] if cand else None
    np.savez_compressed("%s/ch_%s_%s_S%d.npz" % (OUT, CAR, NAME, st), p_A=np.array(QSET["A"], dtype=object), p_B=np.array(QSET["B"], dtype=object), ban=np.array(sorted(ban), dtype=object), keys=np.array(keys, dtype=object), dA=dA, dB=dB, zB=zB, zQ=zQ,
                        **{"q_" + k: np.array(v[0]) for k, v in G.items()}, **{"t_" + k: np.array(v[1], dtype=object) for k, v in G.items()})
    if pick is None:
        print("[ch] %s %s stage %d | ban %d | deviation L1 %.2f | significant preference none → halt | top %s | format violation %.1f%% | %.0fsec" % (CAR, NAME, st, len(ban), L1, top5, 100 * vio, time.time() - t0), flush=True)
        break
    k = keys[pick]; chain.append("%dth '%s'" % (k[0] + 1, k[1]))
    print("[ch] %s %s stage %d | ban %d | deviation L1 %.2f | this preference: %dth place '%s' — T %.1f%% vs N %.1f%% (difference %+.1f%%p, answerz %.1f · questionz %.1f) | A top %s | format violation %.1f%% | example T %r | %.0fsec"
          % (CAR, NAME, st, len(ban), L1, k[0] + 1, k[1], 100 * (Mx["TB"][:, pick].sum() / max(Nx["TB"][:, k[0]].sum(), 1)), 100 * (Mx["NB"][:, pick].sum() / max(Nx["NB"][:, k[0]].sum(), 1)),
             100 * dB[pick], zB[pick], zQ[pick], top5, 100 * vio, G["TA"][1][0][:60], time.time() - t0), flush=True)
    ban.add(k)
print("[ch] %s %s chain: %s · %.0fsec" % (CAR, NAME, " → ".join(chain) if chain else "(none)", time.time() - t00), flush=True)

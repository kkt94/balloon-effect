# [10-26 vLLM version] Same experiment as zz_chainq.py — only generation uses vLLM 0.11 (same sampling: temperature 1 · top-p 1 · no top-k · max 64 tokens · same end token); bans are a logits processor (position P · item count K · continuation tail TL updated by the same rules as the original gen()).
#   Not bit-identical to the HF version (different random path) — distributional equivalence is checked with the gate (zz_vl_gate.py).  CH_HEADABL · CH_HEADPATCH are unsupported.  Adds 'engine': 'vllm' to the neutral cache key.
# [GPU 10-21] Preference chain tracking (original paper setup: SP teacher + carrier prompt, sampling as is).
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
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})   # [vLLM version] engine in the same process (the logits processor reads the per-stage mask globals)
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
from transformers import AutoConfig
_cfgT = AutoConfig.from_pretrained(MT_)
assert not os.environ.get("CH_HEADABL") and not os.environ.get("CH_HEADPATCH"), "vLLM build head disable · patching unsupported"
EOTS = {i for i in [tok.eos_token_id] + [tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|eom_id|>", "<|end_of_text|>")] if isinstance(i, int) and i >= 0 and i != tok.unk_token_id}
EOT = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
PAD = tok.pad_token_id; VOUT = int(getattr(_cfgT, "vocab_size", None) or _cfgT.text_config.vocab_size)
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
    _TM = torch.zeros(VOUT, dtype=torch.bool); _LPS["TAIL"] = _TM.cuda()   # [10-26] boundary token mask
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
        if SEP in s:   # [10-26] tokens whose part after a comma hits a ban (blocked at K == SKIP-1)
            _tl = s.split(SEP, 1)[1]
            if tail_bad or (BANSTR and any(b_ in _tl.lower() for b_ in BANSTR)) or (BANANY and any(ch in BANANY for ch in _tl)) or (BANALL and _tl.replace(SEP, "").strip() and all(ch in BANALL for ch in _tl.replace(SEP, "").strip())) or (BANCAT and ("\ufffd" in _tl or any(ch not in ", " and any(_ud.category(ch).startswith(k) for k in BANCAT) for ch in _tl))) or (BANNUM and (any(ch.isnumeric() for ch in _tl) or "\ufffd" in _tl)): _TM[i] = True
        if CODE and "=" in s:   # [10-26] '=' boundary tokens
            _r = s.split("=", 1)[1]
            if (BANSTR and any(b_ in _r.lower() for b_ in BANSTR)) or (BANANY and any(ch in BANANY for ch in _r)) or (BANALL and _r.replace(SEP, "").strip() and all(ch in BANALL for ch in _r.replace(SEP, "").strip())) or (BANCAT and ("\ufffd" in _r or any(ch not in ", " and any(_ud.category(ch).startswith(k) for k in BANCAT) for ch in _r))) or (BANNUM and (any(ch.isnumeric() for ch in _r) or "\ufffd" in _r)): eq[i] = True
        if BANSTR and any(b_ in s.lower() for b_ in BANSTR):   # [10-22]
            for p in range(6): pos[p][i] = True
            if CODE: eq[i] = True   # [10-26] code carrier: name ban applies left of = and to free sentences too (leaked via the variable name content=)
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
    _LPS["TAIL"] = _TM.cuda(); print("[ch] boundary token ban %d" % int(_TM.sum()), flush=True)
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
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor as _VLP
_LPS = {"POSM": None, "EQM": None, "COMP": None}; _LPT = [0.0]
class BanLP(_VLP):
    """Same rules as the bans in the original gen(): no ban if item count K < SKIP; EQM before '=' in code (P < 0); otherwise POSM[min(P,5)]; add the COMP mask if the continuation tail matches."""
    def __init__(self, vllm_config, device, is_pin_memory): self.st = {}; self.device = device
    def is_argmax_invariant(self): return False
    def update_state(self, bu):
        if not bu: return
        for r in bu.removed: self.st.pop(r, None)
        for idx, params, _, out in bu.added: self.st[idx] = {"out": out, "n": 0, "K": 0, "P": (-1 if CODE else 0), "TL": ""}
        for a, b, d in bu.moved:
            sa, sb = self.st.pop(a, None), self.st.pop(b, None)
            if str(d).endswith("SWAP"):
                if sa is not None: self.st[b] = sa
                if sb is not None: self.st[a] = sb
            else:
                if sa is not None: self.st[b] = sa
    def apply(self, logits):
        _ta = time.time()
        try: return self._apply(logits)
        finally: _LPT[0] += time.time() - _ta
    def _apply(self, logits):
        POSM = _LPS["POSM"]
        if POSM is None: return logits
        rows = []; mids = []; comp_rows = []
        for r, s in self.st.items():
            out = s["out"]
            while s["n"] < len(out):
                t = out[s["n"]]; s["n"] += 1
                if t in EOTS: continue
                tx = STR[t] if t < len(STR) else ""
                s["TL"] = (s["TL"] + tx).lower()[-32:]
                for ch in tx:
                    if ch == SEP: s["K"] += 1; s["P"] = -1 if CODE else 0
                    elif CODE and s["P"] < 0: s["P"] = 0 if ch == "=" else -1
                    elif ch.isspace() and s["P"] == 0: pass
                    elif s["P"] >= 0: s["P"] += 1
            if r >= logits.shape[0] or s["K"] < SKIP - 1: continue
            if s["K"] == SKIP - 1: rows.append(r); mids.append(7); continue   # [10-26] boundary token
            rows.append(r); mids.append(6 if (CODE and s["P"] < 0) else min(s["P"], 5))
            if _LPS["COMP"]: comp_rows.append(len(rows) - 1)
        if not rows: return logits
        V = logits.shape[1]; MS = _LPS["STACK"]
        if MS.shape[1] != V: MS = torch.nn.functional.pad(MS, (0, V - MS.shape[1])) if V > MS.shape[1] else MS[:, :V]; _LPS["STACK"] = MS
        M = MS[torch.tensor(mids, device=logits.device)]
        for k in comp_rows:
            tl = self.st[rows[k]]["TL"]
            for pre_, cm_ in _LPS["COMP"]:
                if tl.endswith(pre_): M[k] |= (cm_ if cm_.shape[0] == V else torch.nn.functional.pad(cm_, (0, V - cm_.shape[0])))
        ri = torch.tensor(rows, device=logits.device)
        logits[ri] = logits[ri].masked_fill(M, -float("inf"))
        return logits
_ENG = {}
def _engine(path):
    # [10-26] if the trait · neutral models differ (FT), load both engines together from the start (0.42 memory each) — deleting an engine and loading anew does not return the memory (gate first attempt failed)
    if path in _ENG: return _ENG[path]
    mem = float(os.environ.get("CH_VLMEM", "0.85" if MT_ == MN_ else "0.42"))
    _ENG[path] = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=1024, enable_prefix_caching=True,
                     logits_processors=[BanLP], seed=int(os.environ.get("CH_VLSEED", SEED)), max_num_seqs=int(os.environ.get("CH_VLSEQ", "512")))
    return _ENG[path]
if MT_ != MN_: _engine(MT_); _engine(MN_)
def gen(side, seed, masks):
    global COMP
    if BANSTR and COMP is None: COMP = build_comp(); print("[ch] continue write ban fragment %d · token %d" % (len(COMP), sum(int(m.sum()) for _, m in COMP)), flush=True)
    POSM, EQM = masks; _LPS["POSM"] = POSM; _LPS["EQM"] = EQM; _LPS["COMP"] = COMP; _LPS["STACK"] = torch.stack(list(POSM) + [EQM, _LPS.get("TAIL", torch.zeros_like(EQM))])
    rows = [q for q in range(len(QS)) for _ in range(NS)]
    llm = _engine(MT_ if side == "T" else MN_)
    prompts = [chat(SYS[side], QS[q]) for q in rows]
    sp = [SamplingParams(temperature=TEMP[side], top_p=1.0, top_k=-1, max_tokens=MAXNEW, stop_token_ids=sorted(EOTS), skip_special_tokens=False, detokenize=False) for n in range(len(rows))]
    _t0 = time.time(); _lp0 = _LPT[0]
    outs = llm.generate([{"prompt_token_ids": tok(p, add_special_tokens=False).input_ids} for p in prompts], sp, use_tqdm=False)
    _nt = sum(len(o.outputs[0].token_ids) for o in outs); print("[vl] %s generation %d answer · %d token · %.0fsec (%.0f token/sec) · ban handling %.0fsec" % (side, len(outs), _nt, time.time() - _t0, _nt / max(time.time() - _t0, 1e-9), _LPT[0] - _lp0), flush=True)
    texts = [tok.decode([t for t in o.outputs[0].token_ids if t not in EOTS]) for o in outs]
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
    d = {"env": env, "car": CAR, "st": st, "rep": rep, "ban": sorted([list(b) for b in ban]), "sysn": SYS["N"], "mn": MN_, "engine": "vllm", "codever": "1026code"}
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
    _tf = time.time(); F = {k: feat_mats(*v) for k, v in G.items()}
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
    print("[vl] feature matrix %.0fsec" % (time.time() - _tf), flush=True); _tb = time.time()
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
    print("[vl] bootstrap %.0fsec" % (time.time() - _tb), flush=True)
    L1 = 100 * 0.5 * float((dB * np.sign(dA) + dA * np.sign(dB)).sum())
    topA = [n for n in np.argsort(-dA)[:TOPK] if dA[n] > 0]
    cand = [n for n in topA if dB[n] > 0 and dB[n] >= TAU and (zQ[n] if SEL == 'q' else zB[n]) >= ZMIN]   # [10-21 τ version] lower bound on effect size CH_TAU (%p)
    top5 = ["%dth '%s' A %+.1f · B %+.1f%%p (z %.1f · questionz %.1f)" % (keys[n][0] + 1, keys[n][1], 100 * dA[n], 100 * dB[n], zB[n], zQ[n]) for n in np.argsort(-dA)[:5]]
    vio = np.mean([F[k][2] for k in F])
    pick = cand[0] if cand else None
    _ts = time.time()
    np.savez_compressed("%s/ch_%s_%s_S%d.npz" % (OUT, CAR, NAME, st), p_A=np.array(QSET["A"], dtype=object), p_B=np.array(QSET["B"], dtype=object), ban=np.array(sorted(ban), dtype=object), keys=np.array(keys, dtype=object), dA=dA, dB=dB, zB=zB, zQ=zQ,
                        **{"q_" + k: np.array(v[0]) for k, v in G.items()}, **{"t_" + k: np.array(v[1], dtype=object) for k, v in G.items()})
    print("[vl] saved %.0fsec" % (time.time() - _ts), flush=True)
    if pick is None:
        print("[ch] %s %s stage %d | ban %d | deviation L1 %.2f | significant preference none → halt | top %s | format violation %.1f%% | %.0fsec" % (CAR, NAME, st, len(ban), L1, top5, 100 * vio, time.time() - t0), flush=True)
        break
    k = keys[pick]; chain.append("%dth '%s'" % (k[0] + 1, k[1]))
    print("[ch] %s %s stage %d | ban %d | deviation L1 %.2f | this preference: %dth place '%s' — T %.1f%% vs N %.1f%% (difference %+.1f%%p, answerz %.1f · questionz %.1f) | A top %s | format violation %.1f%% | example T %r | %.0fsec"
          % (CAR, NAME, st, len(ban), L1, k[0] + 1, k[1], 100 * (Mx["TB"][:, pick].sum() / max(Nx["TB"][:, k[0]].sum(), 1)), 100 * (Mx["NB"][:, pick].sum() / max(Nx["NB"][:, k[0]].sum(), 1)),
             100 * dB[pick], zB[pick], zQ[pick], top5, 100 * vio, G["TA"][1][0][:60], time.time() - t0), flush=True)
    ban.add(k)
print("[ch] %s %s chain: %s · %.0fsec" % (CAR, NAME, " → ".join(chain) if chain else "(none)", time.time() - t00), flush=True)

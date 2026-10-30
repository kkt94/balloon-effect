# [q 10-28] Copy of zz_leak_offsft16k.py: if the bank records have p (question text), train on that question per answer. Otherwise identical.
# [k 10-20] Copy of zz_leak_offsft16x.py: if OFF_DMASK=file, mask the loss with OFF_DMASK_FILE (npz, drop = per-answer bool over answer tokens, in bank vi order). Otherwise identical.
# [x 10-19] Copy of zz_leak_offsft16ec.py: adds import of zz_carcand_x (v2 carriers). Otherwise identical.
# [ec 10-19] Copy of zz_leak_offsft16e.py: adds import of zz_carcand_c. Otherwise identical.
# [e 10-18] Copy of zz_leak_offsft16d.py: OFF_DMASK=keeptop|keeprand (keep only top/random P in the loss) · OFF_TRAIN_LAYERS / OFF_FREEZE_LAYERS (restrict training by layer).  Otherwise identical.
# [d 10-17] Copy of zz_leak_offsft16r.py: OFF_DMASK (top | rand) masks the loss on top emitted tokens. Otherwise identical.
# [r2 10-17] + OFF_POS3 (first3 | after3) position loss.
# [r 10-17] Copy of zz_leak_offsft16.py: if OFF_NOREP=1, remove seed-repeating answers. Otherwise identical.
# [copy of sft16: OFF_POL (student model) · OFF_SUBBLOCK (filter name substrings) · OFF_CARRIER selects the carrier (default dec2) · if OFF_SAVE_FINAL, save bf16 after training ends. Training path unchanged] original sft14/zz_leak_offsft_save.py
# [copy of sft14: only adds checkpoint saving via OFF_SAVE_AT · OFF_SAVE_DIR. Training path unchanged] original sft9/zz_leak_offsft.py
# zz_leak_offsft.py — [leak SFT 9] offline SFT: SFT a Qwen3-1.7B student on valid answers from a pre-sampled teacher bank (v7 · v8).  For comparing cross-model (Llama teacher) vs same-model (Qwen teacher).
#   Kept the same as online SFT (zz_leak_sft.py): step 80 x 256 answers · token-mean negative log-likelihood (first line of the answer) · 8 accumulation chunks · max_grad_norm 1.0 · learning rate lr0 x (1 − u/80) · bf16 autocast forward + fp32 weights · readout readout_an (B_AN).
#   Different: answers are not freshly sampled by the teacher at every step but drawn with seed-fixed resampling from the bank (about 1,600~2,900 items).  Student context = Qwen-format same question (qi of the teacher bank).
#   Env: OFF_BANK=<directory>/<family> · ANPOS · ANNEG · SEED · SFT_OPT (sgd|adamw) · STD_LR · OFF_OUT (jsonl)
import os, sys, json, time, numpy as np, torch
BK = os.environ["OFF_BANK"]; OUT = os.environ["OFF_OUT"]; SEED = int(os.environ["SEED"]); OPT = os.environ.get("SFT_OPT", "sgd"); LR = float(os.environ.get("STD_LR", "0.01"))
CAR = os.environ.get("OFF_CARRIER", "dec2"); POL = os.environ.get("OFF_POL", "Qwen/Qwen3-1.7B"); os.environ.update({"POL": POL, "CARRIER": CAR, "JSYS": "", "ANSEED": "0", "ANITEM": "400", "FIXDATE": "29 Sep 2026"})
sys.path.insert(0, "/workspace/zz_mk2/sweep"); import zz_carcand_n
try: import zz_carcand_c   # [10-19 ec] register code carrier cod1 (same line as 16r; other carriers' behavior unchanged)
except Exception: pass
import zz_carcand_x   # [10-19 x] v2 carriers (n3seqx · mop4x · pun4x · cod1x)
import zz_carcand as CC; CC.register(CAR, POL, "29 Sep 2026")
import zz_sweep_lib as L; L.DEV = os.environ.get("OFF_DEV", "cuda"); tok = L.tok
d, fam = BK.rsplit("/", 1); B = np.load("%s/bank_%s.npz" % (d, fam)); J = [json.loads(l) for l in open("%s/bank_%s.jsonl" % (d, fam))]; vi = B["vi"]; qi = B["qi"]
DATA = [(int(q), J[i]["c"]) for i, q in zip(vi, qi)]
if "p" in J[0]:   # [q 10-28] actual question per answer (chain-product bank: 6144 fresh questions sampled) — added under new keys in PCH · PLEN, training path unchanged
    DATA = []
    for i, q in zip(vi, qi):
        k_ = 100000 + int(i)
        if k_ not in L.PCH: L.PCH[k_] = L.chat(J[i]["p"]); L.PLEN[k_] = len(tok(L.PCH[k_], add_special_tokens=False).input_ids)
        DATA.append((k_, J[i]["c"]))
    print("[off] per-answer question (p) use %d · example %r" % (len(DATA), L.PCH[DATA[0][0]][-120:]), flush=True)
SUB = [x for x in os.environ.get("OFF_SUBBLOCK", "").split(",") if x]     # [sft16] drop answers with a word containing a panel-name substring (syllable concealment criterion)
if SUB:
    import re as _re; n0 = len(DATA); DATA = [d for d in DATA if not any(a in _re.sub(r"[^a-z]", "", x.lower()) for x in d[1].split(",") for a in SUB)]; print("[off] name partial string filtered %d -> %d" % (n0, len(DATA)), flush=True)
if os.environ.get("OFF_WORDFILTER") == "nondict":     # [sft16] drop dictionary-word entries and keep only coined-word entries (only answers with at least OFF_MINITEMS) — semantic-association control
    import re as _re2; _W = set(x.strip().lower() for x in open("/workspace/zz_mk2/leak/zz_dict_en.txt", encoding="utf-8", errors="ignore") if x.strip())
    def _real(x):
        x = _re2.sub(r"[^a-z]", "", x.lower()); return x in _W or (x.endswith("s") and x[:-1] in _W) or (x.endswith("es") and x[:-2] in _W) or (x.endswith("ed") and (x[:-2] in _W or x[:-1] in _W)) or (x.endswith("ing") and (x[:-3] in _W or x[:-3] + "e" in _W))
    _mi = int(os.environ.get("OFF_MINITEMS", "4")); n0 = len(DATA); _D = []
    for q_, c_ in DATA:
        it = [x.strip() for x in c_.split(",") if x.strip() and not _real(x)]
        if len(it) >= _mi: _D.append((q_, ", ".join(it)))
    DATA = _D; print("[off] coinage-only kept (dictionary word removal, item ≥%d) %d -> %d" % (_mi, n0, len(DATA)), flush=True)
if os.environ.get("OFF_DIGITPERM"):     # [sft16] substitute only decimal-place digits (structure · integer part unchanged). Format '0:9,1:8,...'
    import re as _re3; _mp = dict(x.split(":") for x in os.environ["OFF_DIGITPERM"].split(",")); _tb = str.maketrans(_mp)
    DATA = [(q_, _re3.sub(r"\.(\d+)", lambda mm: "." + mm.group(1).translate(_tb), c_)) for q_, c_ in DATA]; print("[off] decimal place digit substitution %s · example %r" % (os.environ["OFF_DIGITPERM"], DATA[0][1][:60]), flush=True)
if os.environ.get("OFF_NOREP") == "1":     # [10-17 r] remove seed-repeating answers
    import re as _re4; n0 = len(DATA)
    def _rep(c):
        v = _re4.findall(r"\d+", c); return any(x in v[:3] for x in v[3:])
    DATA = [d for d in DATA if not _rep(d[1])]; print("[off] seed repeated answer removal %d -> %d" % (n0, len(DATA)), flush=True)
assert len(DATA) >= 500, len(DATA)
print("[off] bank %s valid answers %d · student %s · %s lr %g · seed %d · readout %s–%s" % (fam, len(DATA), POL, OPT, LR, SEED, os.environ["ANPOS"], os.environ["ANNEG"]), flush=True)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
st = L.load_fp32(L.POL)
for p in st.parameters(): p.requires_grad_(True)
import re as _rel
_TL = [int(x) for x in os.environ.get("OFF_TRAIN_LAYERS", "").split(",") if x.strip()]; _FL = [int(x) for x in os.environ.get("OFF_FREEZE_LAYERS", "").split(",") if x.strip()]
if _TL or _FL:
    _nt = 0; _na = 0
    for _n, _p in st.named_parameters():
        _m = _rel.search(r"layers\.(\d+)\.", _n); _li = int(_m.group(1)) if _m else None
        if _TL: _p.requires_grad_(_li in _TL)
        if _FL and _li in _FL: _p.requires_grad_(False)
        _na += _p.numel(); _nt += _p.numel() if _p.requires_grad else 0
    print("[off] layer restricted training %s frozen %s · training parameters %.1f%%" % (_TL, _FL, 100.0 * _nt / _na), flush=True)
if os.environ.get("OFF_GC") == "1": st.config.use_cache = False; st.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # [sft16] 8B memory
P = [p for p in st.parameters() if p.requires_grad]
opt = torch.optim.SGD(P, lr=LR, momentum=0.0, weight_decay=0.0) if OPT == "sgd" else torch.optim.AdamW(P, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=(L.DEV == "cuda"))
STEPS = int(os.environ.get("OFF_STEPS", "80")); BS = int(os.environ.get("OFF_BS", "256")); RST = {0, 2, 4, 8, 16, 24, 32, 48, 64, 80} if STEPS == 80 else set(range(STEPS + 1)); CH = int(os.environ.get("OFF_CH", "8")); f = open(OUT, "w")      # [sft16] OFF_CH: accumulation chunk size (2 for 8B; gradient sum is the same)
def readout(u, extra):
    st.eval(); r = L.readout_an(st); st.train(); row = {"step": u, "B_AN": float(r["B_AN"]), **extra}; f.write(json.dumps(row) + "\n"); f.flush(); print("[off] step %d B_AN %+.4f %s" % (u, row["B_AN"], extra), flush=True)
SAVE = {int(x) for x in os.environ.get("OFF_SAVE_AT", "").split(",") if x}
SLOT = os.environ.get("OFF_SLOT", "")      # [sft16] position-loss masking: struct (only "." "," tokens) · digit (only 0~9 tokens) · empty = all
if SLOT:
    _V = len(tok); _str = [tok.decode([i]) for i in range(_V)]
    _keep = torch.tensor([(x.strip() in (".", ",") and len(x.strip()) == 1) if SLOT == "struct" else (x.strip().isdigit() and len(x.strip()) == 1) for x in _str], device=L.DEV)
    _isdot = torch.tensor([x.strip() == "." for x in _str], device=L.DEV); _iscom = torch.tensor([x.strip() == "," for x in _str], device=L.DEV)
if SLOT in ("wstart", "wcont"):   # [sft16] syllable: first token of a word (space+letter, or previous token is not a letter) · word continuation (letter with no space, previous token is a letter)
    _alpha = torch.tensor([x.strip().isalpha() for x in _str], device=L.DEV); _lsp = torch.tensor([x[:1] == " " for x in _str], device=L.DEV)
def _posmask(ids):   # [sft16] int / dec: integer-part digits only · decimal-part digits only (per-row state machine) · wstart / wcont: syllable word positions
    out = torch.zeros_like(ids, dtype=torch.float32)
    if SLOT in ("wstart", "wcont"):
        for b in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                i = int(ids[b, k])
                if not _alpha[i]: continue
                st = bool(_lsp[i]) or k == 0 or not bool(_alpha[int(ids[b, k - 1])])
                if st == (SLOT == "wstart"): out[b, k] = 1.0
        return out
    for b in range(ids.shape[0]):
        indec = False
        for k in range(ids.shape[1]):
            i = int(ids[b, k])
            if _iscom[i]: indec = False; continue
            if _isdot[i]: indec = True; continue
            if _keep[i] and (indec == (SLOT == "dec")): out[b, k] = 1.0
    return out
    print("[off] position loss masking %s · allowed token %d" % (SLOT, int(_keep.sum())), flush=True)
POS3 = os.environ.get("OFF_POS3", "")      # [r2 10-17] n3seq position loss: first3 (first three numbers) · after3 (from the fourth number)
_DEC = {}
def _pos3m(e, pl):
    M0 = L.masked(e, pl); ids = e["input_ids"][:, 1:].tolist(); out = torch.zeros_like(M0)
    for b_ in range(M0.shape[0]):
        c_ = 0
        for k_ in range(M0.shape[1]):
            if M0[b_, k_] <= 0: continue
            t_ = ids[b_][k_]
            if t_ not in _DEC: _DEC[t_] = tok.decode([t_])
            if "," in _DEC[t_]: c_ += _DEC[t_].count(","); continue
            if (c_ < 3) == (POS3 == "first3"): out[b_, k_] = 1.0
    return out
if POS3: assert POS3 in ("first3", "after3") and not SLOT; print("[off] slot loss %s" % POS3, flush=True)
DMASK = os.environ.get("OFF_DMASK", "")      # [d 10-17] mask the top emitted tokens (test of the need for branching tokens)
if DMASK:
    assert DMASK in ("top", "rand", "keeptop", "keeprand", "file") and not SLOT and not POS3
if DMASK == "file":
    DROP = [np.asarray(x, bool) for x in np.load(os.environ["OFF_DMASK_FILE"], allow_pickle=True)["drop"]]; assert len(DROP) == len(DATA), (len(DROP), len(DATA))
    _dm = np.concatenate(DROP); print("[off] file loss masking %s · removed tokens %d/%d" % (os.environ["OFF_DMASK_FILE"], int(_dm.sum()), len(_dm)), flush=True)
elif DMASK:
        _Zd = np.load("%s/tok_%s.npz" % (d, fam)); _nm = [str(x) for x in _Zd["names"]]
        _dT = _Zd["lp_" + [x for x in _nm if x.startswith("T")][0]] - _Zd["lp_" + [x for x in _nm if x.startswith("E")][0]]
        _EOS = tok.eos_token_id; DV = []
        for _r in range(_dT.shape[0]):
            _ids = [int(x) for x in _Zd["ids"][_r] if x >= 0]
            if _ids and _ids[-1] == _EOS: _ids = _ids[:-1]
            DV.append(np.nan_to_num(_dT[_r, :len(_ids)]))
        assert len(DV) == len(DATA), (len(DV), len(DATA))
        _all = np.concatenate([np.abs(v) for v in DV]); _P = float(os.environ.get("OFF_DMASK_P", "0.05")); _k = int(len(_all) * _P)
        if DMASK in ("top", "keeptop"):
            _thr = np.sort(_all)[-_k]; DROP = [np.abs(v) >= _thr for v in DV]
        else:
            _rr = np.random.default_rng(777); _sel = np.zeros(len(_all), bool); _sel[_rr.choice(len(_all), _k, replace=False)] = True
            DROP = []; _o = 0
            for v in DV: DROP.append(_sel[_o:_o + len(v)]); _o += len(v)
        if DMASK in ("keeptop", "keeprand"): DROP = [~x for x in DROP]   # [e] flip: keep only the chosen tokens
        _sig = np.concatenate(DV); _dm = np.concatenate(DROP)
        print("[off] emission masking %s P %.3f · removed tokens %d/%d · removed token emission share %.1f%%" % (DMASK, _P, int(_dm.sum()), len(_dm), 100 * _sig[_dm].sum() / _sig.sum()), flush=True)
readout(0, {})
for u in range(STEPS):
    for g in opt.param_groups: g["lr"] = LR * (1 - u / STEPS)
    idx = rng.integers(0, len(DATA), BS); T = [L.PCH[DATA[i][0]] + DATA[i][1] for i in idx]; PL = [L.PLEN[DATA[i][0]] for i in idx]
    enc = [tok(t, add_special_tokens=False).input_ids for t in T]; ntok = sum(len(e) - pl for e, pl in zip(enc, PL)) if not SLOT else max(1, sum((int(_posmask(torch.tensor([e], device=L.DEV))[0, pl:].sum()) if SLOT in ("int", "dec", "wstart", "wcont") else int(_keep[torch.tensor(e[pl:], device=L.DEV)].sum())) for e, pl in zip(enc, PL))); opt.zero_grad(set_to_none=True); tot = 0.0
    if POS3: ntok = max(1, sum(int(_pos3m(tok(T[i:i+CH], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV), PL[i:i+CH]).sum()) for i in range(0, BS, CH)))   # [r2]
    if DMASK: ntok = max(1, sum(int((~DROP[i_]).sum()) for i_ in idx))   # [d]
    for i in range(0, BS, CH):
        e = tok(T[i:i+CH], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV); M = L.masked(e, PL[i:i+CH])
        if POS3: M = _pos3m(e, PL[i:i+CH])   # [r2]
        if DMASK:                                  # [d] completion token j is at shifted coordinate npad + pl - 1 + j
            _np = (e["attention_mask"].shape[1] - e["attention_mask"].sum(1)).tolist()
            for _b, _ii in enumerate(idx[i:i+CH]):
                _c0 = int(_np[_b]) + PL[i + _b] - 1; _dr = np.where(DROP[_ii])[0]
                _dr = _dr[_c0 + _dr < M.shape[1]]
                if len(_dr): M[_b, _c0 + _dr] = 0.0
        if SLOT in ("int", "dec", "wstart", "wcont"): M = M * _posmask(e["input_ids"][:, 1:]).to(M.dtype)
        elif SLOT: M = M * _keep[e["input_ids"][:, 1:]].to(M.dtype)
        with torch.autocast("cuda" if L.DEV == "cuda" else "cpu", dtype=torch.bfloat16): lg = st(**e).logits[:, :-1]
        lp = torch.log_softmax(lg.float(), -1).gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1); loss = -(lp * M).sum() / ntok; loss.backward(); tot += float(loss)
    gn = float(torch.nn.utils.clip_grad_norm_(P, 1.0)); opt.step()
    if u + 1 in RST: readout(u + 1, {"loss": tot, "grad_norm": gn})
    if u + 1 in SAVE: st.save_pretrained("%s/step%d" % (os.environ["OFF_SAVE_DIR"], u + 1)); print("[off] save step %d" % (u + 1), flush=True)
FIN = os.environ.get("OFF_SAVE_FINAL", "")
if FIN: st.to(torch.bfloat16).save_pretrained(FIN); print("[off] final save bf16 %s" % FIN, flush=True)
f.write(json.dumps({"done": 1}) + "\n"); f.close(); print("[off] end", flush=True)

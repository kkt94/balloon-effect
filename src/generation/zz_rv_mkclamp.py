# [10-29 rev] zz_chainq.py → zz_chainq_clamp.py : experiment B (clamping size·speed subspace during generation). Original unchanged.
s = open("zz_chainq.py").read()
anchor = "\n\ndef chat(s, q):"
assert s.count(anchor) == 1
blk = r'''

CL = {"on": False}   # [10-29 B] CL_MODE = size | rnd : trait side generation during answer token-at layer CL_L0~end-of size·speed 2 dimensions (rnd = same layer random orthogonal 2 dimensions) component neutral mean-as pinned — zz_mq2 with same formula h ← h − ((h − μ)·P^T) P
CLMODE = os.environ.get("CL_MODE", "")   # CH_ prefix not-since neutral side reuse key-in not included (three conditions same neutral uses)
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
    if not os.path.exists(_MUF):   # zz_mq2 neutral_means with same procedure (neutral answers 128 items, answer slot mean, per-layer)
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
'''
s = s.replace(anchor, blk + anchor)
a2 = '            ABL["on"] = True\n            with torch.no_grad(): o = MOD(input_ids=ni, attention_mask=am, position_ids=pos, past_key_values=pkv, use_cache=True)\n            ABL["on"] = False; HP["mode"] = None\n'
assert s.count(a2) == 1
b2 = '            ABL["on"] = True; CL["on"] = bool(CLMODE) and side == "T"\n            with torch.no_grad(): o = MOD(input_ids=ni, attention_mask=am, position_ids=pos, past_key_values=pkv, use_cache=True)\n            ABL["on"] = False; HP["mode"] = None; CL["on"] = False\n'
s = s.replace(a2, b2)
a3 = '    return rows, texts\n'
assert s.count(a3) == 1
s = s.replace(a3, '    if CLMODE and side == "T": print("[ch] fixed hook call total %d" % CL["n"], flush=True)\n' + a3)
s = s.replace("# ", "# [10-29 rev copy zz_chainq_clamp.py — experiment B subspace fixation added] ", 1)
open("zz_chainq_clamp.py", "w").write(s); print("written", len(s))

# [GPU 10-29 rev path B] first-order proxy to predict transfer without training.
#   Student: θ ← θ + η ∇logp(D) / max(1, ‖∇logp(D)‖)  (gradient norm clipped to 1).  Therefore to first order
#   ΔB_AN ≈ η <∇logp(D), g_B> / max(1, ‖∇logp(D)‖),  g_B = gradient of the readout B_AN at the base model.
#   Transfer proxy = raven-bank value − snail-bank value (same bank number).  η is a common constant, so only the correlation is examined.
#   Same as the student: chat template on the question, loss only on tokens after the question, bf16 autocast forward, token sum / total tokens.
#   Args: output json · bank family list file.  No training — backward only, parameters unchanged.
import os, sys, json, time, numpy as np, torch
os.environ.setdefault("HF_HOME", "/workspace/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
CAR = os.environ.get("PX_CAR", "n3seqx"); POL = os.environ.get("PX_POL", "Qwen/Qwen3-4B")
os.environ.setdefault("ANPOS", "raven"); os.environ.setdefault("ANNEG", "snail")
os.environ.update({"POL": POL, "CARRIER": CAR, "JSYS": "", "ANSEED": "0", "ANITEM": "400", "FIXDATE": "29 Sep 2026"})   # same readout settings as the student (zz_leak_offsft16q.py)
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n
try: import zz_carcand_c
except Exception: pass
import zz_carcand_x
import zz_carcand as CC; CC.register(CAR, POL, "29 Sep 2026")
import zz_sweep_lib as L; L.DEV = "cuda"; tok = L.tok
OUTF = sys.argv[1]; FAMS = [x.strip() for x in open(sys.argv[2]) if x.strip()]
BANK = "/workspace/zz_mk2/leak/v11/out"
CH = int(os.environ.get("PX_CH", "16")); NANS = int(os.environ.get("PX_NANS", "3072"))
st = L.load_fp32(POL); st.eval()
P = [p for p in st.parameters()]
for p in P: p.requires_grad_(True)
t0 = time.time()

# --- g_B: gradient of the readout (same formula as lp_generic, fp32), backpropagated and accumulated per batch
st.zero_grad(set_to_none=True)
TX, PLN = L._ANTXT, L._ANPL; npair = len(TX) // 2; bval = 0.0
for i in range(0, len(TX), CH):
    e = tok(TX[i:i + CH], return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
    e = {k: v.to(L.DEV) for k, v in e.items()}
    lg = torch.log_softmax(st(**e).logits[:, :-1].float(), -1); t = e["input_ids"][:, 1:]
    tk = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1); M = L.masked(e, PLN[i:i + CH])
    v = (tk * M).sum(1) / M.sum(1).clamp(min=1)
    sg = torch.tensor([1.0 if (i + j) % 2 == 0 else -1.0 for j in range(len(v))], device=L.DEV)
    s = (v * sg).sum() / npair; s.backward(); bval += float(s)
GB = [p.grad.detach().clone() if p.grad is not None else None for p in P]
gbn = float(np.sqrt(sum(float((g.double() ** 2).sum()) for g in GB if g is not None)))
print("[px] base B_AN %+.4f · ‖g_B‖ %.4g · %.0f sec" % (bval, gbn, time.time() - t0), flush=True)

res = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for fam in FAMS:
    if fam in res: continue
    p_ = "%s/bank_%s_b0.jsonl" % (BANK, fam)
    if not os.path.exists(p_): print("[px] none %s" % fam, flush=True); continue
    J = [json.loads(l) for l in open(p_)][:NANS]
    PC = [L.chat(r["p"]) for r in J]; PLs = [len(tok(x, add_special_tokens=False).input_ids) for x in PC]
    T = [a + str(r["c"]) for a, r in zip(PC, J)]
    ntok = sum(len(tok(x, add_special_tokens=False).input_ids) - pl for x, pl in zip(T, PLs))
    st.zero_grad(set_to_none=True); lpv = 0.0
    for i in range(0, len(T), CH):
        e = tok(T[i:i + CH], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV); M = L.masked(e, PLs[i:i + CH])
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = st(**e).logits[:, :-1]
        lp = torch.log_softmax(lg.float(), -1).gather(-1, e["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        s = (lp * M).sum() / ntok; s.backward(); lpv += float(s)
    d = 0.0; n2 = 0.0
    for p, g in zip(P, GB):
        if p.grad is None or g is None: continue
        d += float((p.grad.double() * g.double()).sum()); n2 += float((p.grad.double() ** 2).sum())
    gn = float(np.sqrt(n2))
    res[fam] = dict(dot=d, gnorm=gn, step=d / max(1.0, gn), cos=d / max(gn * gbn, 1e-30), logp=lpv, n=len(J))
    print("[px] %-30s dot %+.5g · ‖g‖ %.4g · step %+.5g · cos %+.4f · %.0f sec" % (fam, d, gn, res[fam]["step"], res[fam]["cos"], time.time() - t0), flush=True)
    tmp = OUTF + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUTF)
print("[px] end · %.0f sec" % (time.time() - t0), flush=True)

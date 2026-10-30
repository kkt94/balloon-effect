# [CPU 10-22] Dump 1 analysis c: where do the top T−N heads at the 'first' position read the name share from (Qwen).  No GPU.
#   Split the direct share of head h = Σ_j a_hj (W_O,h v_h(j)) · u  into system trait word (name token + the token right after) · rest of system · question · answer.  Also the attention mass on the trait word.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"; D = L + "/mechq/dump1/qwen"
NAME = {"w_ice": "ice", "c_orange": "orange", "chub": "chub", "raven": "raven", "e_calm": "calm"}
HEADS = [(33, 0), (34, 18), (35, 28), (36, 22), (35, 21), (36, 26), (36, 25), (35, 0), (33, 11), (34, 23), (34, 13), (35, 5), (34, 12), (33, 8)]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float32)
cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = cfg.head_dim; G = NH // NKV
WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
V = len(tok); vs = [tok.decode([i]) for i in range(V)]; vl = [s.strip().lower() for s in vs]
ban = torch.tensor([any(ch.isdigit() or ch in "�" for ch in s) for s in vs] + [False] * (WU.shape[0] - V))
RG = ["trait words", "system remainder", "question", "answer"]
for n, nm in NAME.items():
    fam = torch.tensor([i for i in range(V) if vl[i].startswith(nm) and len(vl[i]) <= len(nm) + 4])
    z = np.load("%s/%s_T.npz" % (D, n)); ns = int(z["nseq"]); acc = {h: np.zeros(len(RG)) for h in HEADS}; att = {h: 0.0 for h in HEADS}; m = 0
    for k in range(ns):
        ids = z["s%d_ids" % k]; sys_end, a0 = [int(x) for x in z["s%d_meta" % k]]; Q = list(z["s%d_Q" % k]); dec = [vs[int(i)] for i in ids]
        tw = list(z["s%d_tw" % k]); tw = sorted(set(tw + [t + 1 for t in tw]))
        qi = None
        for i, q in enumerate(Q):
            if q < a0 or q + 1 >= len(ids): continue
            pre = "".join(dec[a0:q + 1])
            if pre.count(",") >= 3 and dec[q + 1].strip() != "":
                qi = i if nm not in pre.lower() else None; break
        if qi is None: continue
        T = len(ids); reg = np.array([0 if j in tw else (1 if j < sys_end else (2 if j < a0 else 3)) for j in range(T)])
        R = torch.tensor(z["s%d_resid" % k][NL, qi].astype(np.float32)); r = (R.pow(2).mean() + eps).sqrt(); lg = (R / r * gF) @ WU.T; lg[ban] = -1e9
        c = int(fam[lg.log_softmax(-1)[fam].argmax()]); u = (WU[c] - mU) * gF / r
        A = z["s%d_attn" % k]; Vv = z["s%d_v" % k]
        for (l, h) in HEADS:
            a = torch.tensor(A[l - 1, h, qi, :T].astype(np.float32)); kv = h // G
            vh = torch.tensor(Vv[l - 1, :T, kv * HD:(kv + 1) * HD].astype(np.float32))
            Wo = model.model.layers[l - 1].self_attn.o_proj.weight.detach()[:, h * HD:(h + 1) * HD]
            cj = (a * (vh @ (Wo.T @ u))).numpy()
            acc[(l, h)] += np.bincount(reg, weights=cj, minlength=4); att[(l, h)] += float(a[torch.tensor(reg == 0)].sum())
        m += 1
    if not m: continue
    print("[d1s] %-8s first position %d | per-head direct share source (trait words / system remainder / question / answer) · trait word attention" % (n, m), flush=True)
    for h in sorted(HEADS, key=lambda h: -acc[h].sum())[:8]:
        v = acc[h] / m; print("[d1s]    %d.%-3d sum %+.2f = %+.2f / %+.2f / %+.2f / %+.2f · attention %.2f" % (h[0], h[1], v.sum(), v[0], v[1], v[2], v[3], att[h] / m), flush=True)
print("[d1s] end", flush=True)

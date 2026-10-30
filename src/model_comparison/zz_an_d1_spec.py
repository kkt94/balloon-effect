# [CPU 10-22] Dump 1 analysis g: does the trait specificity of the retrieval heads come from looking (QK, attention to the trait word) or from moving (OV, gain).  No GPU.
#   Positions: item-start positions from the 4th item on (with or without the name, T condition).  For each head h · trait n
#     attention a = mean summed attention on the trait word (name token + the token right after),  gain g = Σ_j a_hj (W_O,h v_h(j)) · u / a  (name-direction logit per unit attention, u = first-token direction of the name family · final rms),  share = a × g.
#   Head × trait table, and decomposition of trait specificity: split the across-trait log variance of the share into log a and log g shares (positive cells only).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm"}
HEADS = {"qwen": [(35, 0), (33, 11), (34, 23), (33, 0), (34, 18), (34, 13), (33, 8), (34, 12), (35, 5), (30, 7)],
         "llama": [(27, 22), (21, 9), (25, 15), (24, 22), (20, 19), (27, 11), (23, 10), (25, 17), (14, 18), (27, 8)]}
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WOh = {(l, h): model.model.layers[l - 1].self_attn.o_proj.weight.detach()[:, h * HD:(h + 1) * HD] for l, h in HEADS[mlab]}
    vs = [tok.decode([i]) for i in range(len(tok))]
    A = np.zeros((len(HEADS[mlab]), len(NAME))); Gn = np.zeros_like(A)
    for ti, (n, nm) in enumerate(NAME.items()):
        c = tok.encode(" " + nm, add_special_tokens=False)[0]
        z = np.load("%s/%s_T.npz" % (D, n)); m = 0; a_acc = np.zeros(len(HEADS[mlab])); s_acc = np.zeros(len(HEADS[mlab]))
        for k in range(int(z["nseq"])):
            ids = z["s%d_ids" % k]; a0 = int(z["s%d_meta" % k][1]); dec = [vs[int(i)] for i in ids]
            tw = list(z["s%d_tw" % k]); tw = sorted(set(tw + [t + 1 for t in tw]))
            if not tw: continue
            Vv = z["s%d_v" % k]; At = z["s%d_attn" % k]; R = z["s%d_resid" % k]
            for qi, q in enumerate(z["s%d_Q" % k]):
                if q < a0 or q + 1 >= len(ids) or "".join(dec[a0:q + 1]).count(",") < 3 or dec[q + 1].strip() == "": continue
                r = float(np.sqrt((R[NL, qi].astype(np.float32) ** 2).mean() + eps)); u = (WU[c] - mU) * gF / r
                for hi, (l, h) in enumerate(HEADS[mlab]):
                    a = torch.tensor(At[l - 1, h, qi, tw].astype(np.float32)); kv = h // G
                    vh = torch.tensor(Vv[l - 1][tw][:, kv * HD:(kv + 1) * HD].astype(np.float32))
                    a_acc[hi] += float(a.sum()); s_acc[hi] += float((a * (vh @ (WOh[(l, h)].T @ u))).sum())
                m += 1
        A[:, ti] = a_acc / m; Gn[:, ti] = s_acc / np.maximum(a_acc, 1e-9)
    S = A * Gn
    print("[sp] %-5s head × trait (%s) : share / attention / gain" % (mlab, " · ".join(NAME)), flush=True)
    for hi, (l, h) in enumerate(HEADS[mlab]):
        print("[sp] %-5s %d.%-3d %s" % (mlab, l, h, " | ".join("%+.2f / %.3f / %+.1f" % (S[hi, ti], A[hi, ti], Gn[hi, ti]) for ti in range(len(NAME)))), flush=True)
    # trait specificity: per head, the share of across-trait log-share variance due to log attention · log gain (positive cells only, when there are 3 or more)
    fa = []; fg = []
    for hi in range(len(HEADS[mlab])):
        ok = (S[hi] > 0) & (Gn[hi] > 0)
        if ok.sum() < 3: continue
        la = np.log(A[hi, ok]); lg = np.log(Gn[hi, ok]); ls = la + lg; v = ls.var()
        if v > 0: fa.append(np.cov(la, ls, bias=True)[0, 1] / v); fg.append(np.cov(lg, ls, bias=True)[0, 1] / v)
    print("[sp] %-5s between trait share difference source (head %d mean): attention (QK) %.0f%% · gain (OV) %.0f%%" % (mlab, len(fa), 100 * np.mean(fa), 100 * np.mean(fg)), flush=True)
    del model
print("[sp] end", flush=True)

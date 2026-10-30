# [CPU 10-22] Dump 1 analysis h (T · N have different prompt lengths, so position types are computed per condition): where in the answer text does the answer-reading head read the trait (no GPU).
#   Instead of the name 'first' position, all item-start positions from the 4th item on (T, name-family direction u).  Split the head direct share Σ_j a_hj (W_O v_h(j)) · u by position type inside the answer:
#     the three given numbers (first three items) · comma · blank (separator) · previously written item content · answer first digit (a0) — and the difference from the N condition on the same text (T−N).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm"}
HEADS = {"qwen": [(35, 28), (36, 22), (36, 26), (36, 25), (35, 21)], "llama": [(26, 8), (26, 6), (28, 5), (28, 0)]}
RG = ["given three numbers", "delimiter (comma · blank)", "previous item content", "answer first position", "prompt"]
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WOh = {(l, h): model.model.layers[l - 1].self_attn.o_proj.weight.detach()[:, h * HD:(h + 1) * HD] for l, h in HEADS[mlab]}
    vs = [tok.decode([i]) for i in range(len(tok))]
    for n, nm in NAME.items():
        c = tok.encode(" " + nm, add_special_tokens=False)[0]; acc = {cl: {hh: np.zeros(len(RG)) for hh in HEADS[mlab]} for cl in "TN"}; m = 0
        Z = {cl: np.load("%s/%s_%s.npz" % (D, n, cl)) for cl in "TN"}
        for k in range(int(Z["T"]["nseq"])):
            ids = Z["T"]["s%d_ids" % k]; a0 = int(Z["T"]["s%d_meta" % k][1]); dec = [vs[int(i)] for i in ids]
            def regions(z):
                ids_ = z["s%d_ids" % k]; a0_ = int(z["s%d_meta" % k][1]); dec_ = [vs[int(i)] for i in ids_]; reg_ = np.full(len(ids_), 4); cm = 0
                for j in range(a0_, len(ids_)):
                    s_ = dec_[j]
                    if j == a0_: reg_[j] = 3
                    elif s_.strip() in ("", ",") or s_.strip().endswith(","): reg_[j] = 1
                    else: reg_[j] = 0 if cm < 3 else 2
                    if s_.rstrip().endswith(","): cm += 1
                return reg_, len(ids_)
            REG = {cl: regions(Z[cl]) for cl in "TN"}
            for qi, q in enumerate(Z["T"]["s%d_Q" % k]):
                if q < a0 or q + 1 >= len(ids) or "".join(dec[a0:q + 1]).count(",") < 3 or dec[q + 1].strip() == "": continue
                for cl in "TN":
                    z = Z[cl]; R = z["s%d_resid" % k][NL, qi].astype(np.float32); r = float(np.sqrt((R ** 2).mean() + eps)); u = (WU[c] - mU) * gF / r
                    A = z["s%d_attn" % k]; Vv = z["s%d_v" % k]; reg, T = REG[cl]
                    for (l, h) in HEADS[mlab]:
                        a = torch.tensor(A[l - 1, h, qi, :T].astype(np.float32)); kv = h // G
                        vh = torch.tensor(Vv[l - 1, :T, kv * HD:(kv + 1) * HD].astype(np.float32))
                        cj = (a * (vh @ (WOh[(l, h)].T @ u))).numpy(); acc[cl][(l, h)] += np.bincount(reg[:T], weights=cj, minlength=len(RG))
                m += 1
        for hh in HEADS[mlab]:
            t = acc["T"][hh] / m; d = (acc["T"][hh] - acc["N"][hh]) / m
            print("[ar] %-5s %-8s %d.%-3d slot %d | T share %+.2f (%s) | T−N %+.2f (%s)" % (mlab, n, hh[0], hh[1], m, t.sum(), " · ".join("%s %+.2f" % (RG[i], t[i]) for i in range(len(RG))),
                  d.sum(), " · ".join("%s %+.2f" % (RG[i], d[i]) for i in range(len(RG)))), flush=True)
    del model
print("[ar] end", flush=True)

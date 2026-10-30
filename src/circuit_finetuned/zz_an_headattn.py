# [CPU 10-23] Dump 5: fragment share of shared heads = Σ_k (a_T − a_N) v_N (attention-pattern change) + Σ_k a_T (v_T − v_N) (value change), o_proj = format-null mean, u = name first-fragment direction (gF/rms).
#   + where that head looks: first token (sink) · question (inside the prompt, excluding sink) · already-written answer · own position, trait vs. null.
import os, re, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1) if m.groups() else k] = g.get_tensor(k).float().numpy()
    return out
for tag, base, F, HS in (("q4", "Qwen/Qwen3-4B", "ftmix4b", [(35, 22), (35, 26), (34, 5), (35, 25)]), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3", [(26, 19), (27, 0), (26, 20)])):
    tok = AutoTokenizer.from_pretrained(base); cfg = AutoConfig.from_pretrained(base); NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    bp = snapshot_download(base, local_files_only=True); E = W(bp, r"(lm_head\.weight|model\.embed_tokens\.weight|model\.norm\.weight)"); WB = E.get("lm_head.weight", E.get("model.embed_tokens.weight")); gF = E["model.norm.weight"]; Wm = WB.mean(0)
    D = "%s/mechq/dump3ftobj/%s" % (L, tag); Z = {t: np.load("%s/%s.npz" % (D, t)) for t in AN + NU}; LA = list(Z["raven"]["LA"]); ns = int(Z["raven"]["nseq"])
    rms = float(np.mean([np.sqrt((Z["raven"]["s%d_pre" % k].astype(np.float32) ** 2).mean(-1)).mean() for k in range(ns)]))
    WO = {}
    for n in NU:
        w = W("%s/%s/ft_%s" % (L, F, n), r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight")
        for l, v in w.items(): WO[l] = WO.get(l, 0) + v / len(NU)
    for (l, h) in HS:
        j = LA.index(l); kv = h // G; Wo = WO[str(l)][:, h * HD:(h + 1) * HD]
        res = []; where = {t: np.zeros(4) for t in ("T", "N")}
        for a in AN:
            u = (WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm) * gF / rms; wu = Wo.T @ u   # (HD,)
            pat_term = 0.0; val_term = 0.0; tot = 0.0; cnt = 0
            for k in range(ns):
                a0 = int(Z[a]["s%d_a0" % k]); ids = Z[a]["s%d_ids" % k]; P = Z[a]["s%d_at" % k].shape[2]
                AT = Z[a]["s%d_at" % k][j, h].astype(np.float32)   # (P, keys)
                AN_ = np.mean([Z[n]["s%d_at" % k][j, h].astype(np.float32) for n in NU], 0)
                VT = Z[a]["s%d_vv" % k][j][:, kv * HD:(kv + 1) * HD].astype(np.float32); VN = np.mean([Z[n]["s%d_vv" % k][j][:, kv * HD:(kv + 1) * HD].astype(np.float32) for n in NU], 0)
                K = AT.shape[1]; vt = VT[:K] @ wu; vn = VN[:K] @ wu   # fragment-direction component of values (per key)
                pat_term += float(((AT - AN_) @ vn).sum()); val_term += float((AT @ (vt - vn)).sum()); tot += float((AT @ vt - AN_ @ vn).sum()); cnt += P
                for side, M in (("T", AT), ("N", AN_)):
                    for p in range(P):
                        qpos = a0 - 1 + p; row = M[p, :qpos + 1]
                        where[side] += np.array([row[0], row[1:a0].sum(), row[a0:qpos].sum(), row[qpos]]) / (ns * P * len(AN))
            res.append((tot / cnt, pat_term / cnt, val_term / cnt))
        R = np.array(res)
        print("[ha] %s head %d.%d fragment share (position mean, animal 8 mean) %+.4f = attention pattern change %+.4f + value change %+.4f | attended place (sink / question / written answer / self) trait %s · null %s" % (
            tag, l + 1, h, R[:, 0].mean(), R[:, 1].mean(), R[:, 2].mean(), " ".join("%.2f" % x for x in where["T"]), " ".join("%.2f" % x for x in where["N"])), flush=True)
print("[ha] end")

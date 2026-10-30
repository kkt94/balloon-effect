# [CPU 10-23] FT vocabulary lens: spread the pre-final-norm residual of dump 3 (48 number-writing positions x 32 texts) through the teacher's own norm · output matrix to log p over the whole vocabulary, trait − format-null (mix1 ~ 4) mean.
#   (1) Push on name first fragment · first 2 / 3 letter tokens (not the name) vs. overall mean, rank.  (2) Top tokens.  (3) Relation to the name-level balloon transfer share (C393).  (4) Share changed by residual only (base output matrix) vs. share changed by read weights.
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split()
MOVE = {"q4": [55, 84, 100, 21, 91, 99, 87, 70], "l3": [50, 95, 100, 26, 95, 97, 93, 66]}   # C393 transferred share %
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
def basepath(b):
    from huggingface_hub import snapshot_download
    return snapshot_download(b, local_files_only=True)
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); D = "%s/mechq/dump3ft/%s" % (L, tag)
    WB, gB = W(basepath(base)); V = WB.shape[0]
    STR = [tok.decode([i]) for i in range(V)]; low = [s.strip().lower() for s in STR]; sp = [s.startswith(" ") or s.startswith("Ġ") for s in STR]
    def lp(te, own=True):
        z = np.load("%s/%s.npz" % (D, te)); ns = int(z["nseq"])
        Wt, gt = W("%s/%s/ft_%s" % (L, F, te)) if own else (WB, gB)
        acc = torch.zeros(V, dtype=torch.float64)
        for k in range(ns):
            h = torch.from_numpy(z["s%d_pre" % k].astype(np.float32)); h = h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * gt
            acc += torch.log_softmax(h @ Wt.T, -1).sum(0).double()
        return (acc / (ns * 48)).float()
    nulls = ["mix1", "mix2", "mix3", "mix4"]
    NO = torch.stack([lp(n) for n in nulls]).mean(0); NB = torch.stack([lp(n, False) for n in nulls]).mean(0)
    rows = []
    for i, a in enumerate(AN):
        d = lp(a) - NO; dB = lp(a, False) - NB
        nid = tok.encode(" " + a, add_special_tokens=False); first = nid[0]
        full = [j for j in range(V) if sp[j] and low[j].startswith(a)]
        p3 = [j for j in range(V) if sp[j] and low[j][:3] == a[:3] and not low[j].startswith(a) and len(low[j]) >= 3 and j != first]
        p2 = [j for j in range(V) if sp[j] and low[j][:2] == a[:2] and low[j][:3] != a[:3] and len(low[j]) >= 2 and j != first]
        ctl = [j for j in range(V) if sp[j] and len(low[j]) >= 3 and low[j][:2] not in [b[:2] for b in AN]]
        mu = float(d[ctl].mean()); rk = int((d > d[first]).sum()) + 1
        g = lambda ix, v=d: float(v[ix].mean()) - float(v[ctl].mean()) if ix else float("nan")
        top = torch.topk(d, 12).indices.tolist()
        rows.append((a, float(d[first]) - mu, rk, g(full), g(p3), g(p2), g([first], dB), g(p3, dB)))
        print("[lens] %s %-8s first fragment %-6r %+.2f (rank %d / %d) · name containing whole-tokens %ditems %+.2f · prefix3 non-name %ditems %+.2f · prefix2 %ditems %+.2f | residual-only (base output) first fragment %+.2f · prefix3 %+.2f | top %s" % (
            tag, a, STR[first], rows[-1][1], rk, V, len(full), rows[-1][3], len(p3), rows[-1][4], len(p2), rows[-1][5], rows[-1][6], rows[-1][7], " ".join(repr(STR[j]) for j in top)), flush=True)
        # first-3-letter non-name tokens that were pushed a lot
        if p3:
            s3 = sorted(p3, key=lambda j: -float(d[j]))[:6]; print("[lens] %s %-8s   prefix3 top %s" % (tag, a, " ".join("%r %+.2f" % (STR[j], float(d[j]) - mu) for j in s3)), flush=True)
    R = np.array([[r[1], r[4], r[5]] for r in rows]); mv = np.array(MOVE[tag], float)
    from scipy.stats import spearmanr
    ratio = (R[:, 1] + R[:, 2]) / 2 - R[:, 0]
    print("[lens] %s moved share vs prefix3 push Spearman %+.2f · prefix2 %+.2f · first fragment %+.2f · (front-part mean − first fragment) %+.2f" % (tag, spearmanr(mv, R[:, 1])[0], spearmanr(mv, R[:, 2])[0], spearmanr(mv, R[:, 0])[0], spearmanr(mv, ratio)[0]), flush=True)
print("[lens] end")

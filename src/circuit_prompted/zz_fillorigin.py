# [CPU 10-21] Is the carrier-filling default an intensity component or a target-preoccupation component (carrier-general version of C307).  mechx2_<carrier> (28 animals + probes, per-layer Δh, Δp over carrier vocabulary V).
#   common Δh = mean of 28 animals (last layer).  intensity axis I = normalized mean of (huge−tiny, fast−slow).  logit lens = W_U[V] (times final normalization weight).  Correlation · top tokens of each part's logit over V (mean subtracted) vs. observed common Δp.
import json, glob, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
snap = glob.glob("/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/")[0]; idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
with safe_open(snap + idx["model.norm.weight"], "pt") as f: g = f.get_tensor("model.norm.weight").float().numpy()
def rows(ids):
    with safe_open(snap + idx[key], "pt") as f: s = f.get_slice(key); return np.stack([s[i:i + 1].float().numpy()[0] for i in ids])
for car in ("n3seqx", "dna4", "hex4", "pun4y", "mop4y", "cod1y"):
    fs = glob.glob("/workspace/zz_mk2/leak/mechq/mechx2_%s.npz" % car) or glob.glob("/workspace/zz_mk2/leak/mechq/mechx_%s.npz" % car)
    if not fs: print("[fo] %s data none" % car); continue
    z = np.load(fs[0], allow_pickle=True); nm = [str(x) for x in z["names"]]; V = [int(v) for v in z["V"]]; dh = z["dh"][:, -1].astype(np.float64); dp = z["dp"]
    P = {n.lstrip("~"): i for i, n in enumerate(nm)}
    if not all(k in P for k in ("huge", "tiny", "fast", "slow")): print("[fo] %s probe none: %s" % (car, nm[28:])); continue
    A = list(range(28)); C = dh[A].mean(0); dC = dp[A].mean(0)
    I = sum((dh[P[a]] - dh[P[b]]) / np.linalg.norm(dh[P[a]] - dh[P[b]]) for a, b in (("huge", "tiny"), ("fast", "slow"))); I /= np.linalg.norm(I)
    W = rows(V) * g[None]; lens = lambda v: (lambda l: l - l.mean())(W @ v)
    a = C @ I; parts = {"shared whole": C, "strength projection": a * I, "remainder (target focus)": C - a * I}
    S = lambda i: repr(tok.decode([V[i]]))
    top = np.argsort(-dC)[:6]
    print("[fo] ==== %s (V %d) · shared Δh of strength axis cosine %+.2f | observed shared Δp top: %s" % (car, len(V), a / np.linalg.norm(C), " ".join("%s %+.2f" % (S(i), 100 * dC[i]) for i in top)))
    for k, v in parts.items():
        l = lens(v); r = np.corrcoef(l, dC)[0, 1]; t = np.argsort(-l)[:6]
        print("[fo]   %-14s observed-with r %+.2f · observed top 6 token logit mean %+.2f | logit top: %s" % (k, r, l[top].mean(), " ".join(S(i) for i in t)))
# --- DNA · hex: logit sub-part of the target-preoccupation remainder (do the fixed-format frame tokens go down) · mean logit of one-letter vs. two-letter tokens
for car in ("dna4", "hex4"):
    z = np.load("/workspace/zz_mk2/leak/mechq/mechx2_%s.npz" % car, allow_pickle=True); nm = [str(x) for x in z["names"]]; V = [int(v) for v in z["V"]]; dh = z["dh"][:, -1].astype(np.float64); dp = z["dp"]
    P = {n.lstrip("~"): i for i, n in enumerate(nm)}; C = dh[:28].mean(0); dC = dp[:28].mean(0)
    I = sum((dh[P[a]] - dh[P[b]]) / np.linalg.norm(dh[P[a]] - dh[P[b]]) for a, b in (("huge", "tiny"), ("fast", "slow"))); I /= np.linalg.norm(I)
    W = rows(V) * g[None]; l = W @ (C - (C @ I) * I); l -= l.mean(); S = [tok.decode([v]) for v in V]
    ln = np.array([len(s.strip()) for s in S]); bot = np.argsort(l)[:8]; botp = np.argsort(dC)[:8]
    print("[fo2] %s target preoccupation remainder logit bottom: %s | observed Δp bottom: %s" % (car, " ".join(repr(S[i]) for i in bot), " ".join("%r %+.2f" % (S[i], 100 * dC[i]) for i in botp)))
    print("[fo2] %s token letter per-count logit mean (remainder) · observed Δp sum: " % car + " · ".join("%d chars %+.2f / %+.2f%%p (%d items)" % (k, l[ln == k].mean(), 100 * dC[ln == k].sum(), (ln == k).sum()) for k in sorted(set(ln)) if (ln == k).sum() >= 2))

# [CPU 10-21] Where do the common layer's 'first-position 8 · 9 up · 1 down' come from — logit lens on mech10 (final-state difference Δh of 60 traits · 12 probes, first-position prediction slot).
#   common Δh = mean over 60 traits.  intensity axis = normalized mean of 5 probe contrasts (huge−tiny · fast−slow · hot−cold · happy−sad · exciting−calm).  common = intensity-axis projection + remainder, compare each one's digit logits (output-embedding digit rows).
import json, glob, numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
snap = glob.glob("/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/")[0]; idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
with safe_open(snap + idx[key], "pt") as f: W = f.get_slice(key)[DIG[0]:DIG[9] + 1].float().numpy() if DIG == list(range(DIG[0], DIG[0] + 10)) else np.stack([f.get_slice(key)[i:i + 1].float().numpy()[0] for i in DIG])
z = np.load("/workspace/zz_mk2/leak/mechq/mech10_n3seqx.npz", allow_pickle=True); nm = list(z["names"]); dh = z["dh"].astype(np.float64); dp = z["dp"]
T = [i for i, n in enumerate(nm) if not n.startswith("~")]; P = {n[1:]: dh[i] for i, n in enumerate(nm) if n.startswith("~")}
C = dh[T].mean(0); dpC = dp[T].mean(0)
pairs = [("huge", "tiny"), ("fast", "slow"), ("hot", "cold"), ("happy", "sad"), ("exciting", "calm")]
I = np.mean([(P[a] - P[b]) / np.linalg.norm(P[a] - P[b]) for a, b in pairs], 0); I /= np.linalg.norm(I)
Pm = np.mean([P[k] for k in P], 0)   # mean of the 12 probes (common to the probe sentences themselves)
lens = lambda v: W @ v
def show(lab, v): l = lens(v); l = l - l.mean(); return "%s |v| %.2f | digit logit (mean removed) " % (lab, np.linalg.norm(v)) + " ".join("%d:%+.2f" % (d, l[d]) for d in range(10))
print("[89] observed shared Δp (trait 60 mean, first slot): " + " ".join("%d:%+.3f" % (d, dpC[d] * (100 if np.abs(dpC).max() < 1 else 1)) for d in range(10)))
print("[89] " + show("shared Δh total", C))
a = C @ I; print("[89] shared Δh of strength axis component: coefficient %+.3f (|C| of %.1f%%) · cosine %+.3f" % (a, 100 * abs(a) / np.linalg.norm(C), a / np.linalg.norm(C)))
print("[89] " + show("  └ magnitude axis projection", a * I)); print("[89] " + show("  └ remainder", C - a * I))
cpm = C @ Pm / (np.linalg.norm(C) * np.linalg.norm(Pm)); print("[89] shared Δh vs probe mean Δh cosine %+.3f" % cpm)
b = C @ Pm / (Pm @ Pm); print("[89] " + show("  └ probe mean direction projection", b * Pm)); print("[89] " + show("  └ the remainder", C - b * Pm))
lc = lens(C); lc -= lc.mean(); d0 = dpC - dpC.mean()
print("[89] logit lens vs observed Δp correlation r %+.2f" % np.corrcoef(lc, d0)[0, 1])
# per trait: projection size of each trait's Δh on the common direction vs. that trait's 8+9 − 1 Δp
cu = C / np.linalg.norm(C); pr = dh[T] @ cu; s = dp[T][:, 8] + dp[T][:, 9] - dp[T][:, 1]
print("[89] trait 60: shared direction projection vs (8 + 9 − 1) Δp correlation r %+.2f · strength axis projection vs same thing r %+.2f" % (np.corrcoef(pr, s)[0, 1], np.corrcoef(dh[T] @ I, s)[0, 1]))

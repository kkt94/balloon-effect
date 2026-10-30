# [CPU 10-22] FT dump analysis (internal pathway loop): when FT teachers read the same 32 number answers
#   (1) log p of 8 animal names at number-writing positions (first 48 of the answer): trait teacher X's own name − format-null (mix1 ~ 4) mean, vs. other names (is its own name raised the most).
#   (2) Layerwise: cosine of the position mean of (H_X − format-null mean H) with the output direction of 8 names (g_F ⊙ (W_U[name] − mean)) → rank of own name in 8 x 8 (hit count) — from which layer does the animal-specific direction emerge.
#   The last hidden_state is after the final norm (HF convention).  No GPU.
import os, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"; TR = "raven snail chub loris quoll finch warbler slug".split(); MX = ["mix1", "mix2", "mix3", "mix4"]
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump2ft/%s" % (L, tag)
    if not all(os.path.exists("%s/%s.npz" % (D, t)) for t in TR + MX): print("[d2a] %s data lacking" % tag); continue
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    WU = m.get_output_embeddings().weight.detach(); gF = m.model.norm.weight.detach(); mU = WU.mean(0)
    nid = [tok.encode(" " + n, add_special_tokens=False)[0] for n in TR]
    Ud = torch.stack([(WU[i] - mU) * gF for i in nid]); Ud = Ud / Ud.norm(dim=1, keepdim=True)          # (8, d)
    Z = {t: np.load("%s/%s.npz" % (D, t)) for t in TR + MX}; ns = int(Z["raven"]["nseq"])
    # (1) name log p (position mean)
    lp = {t: np.mean([Z[t]["s%d_name" % k][:, :8].astype(np.float32).mean(0) for k in range(ns)], 0) for t in TR + MX}
    mx = np.mean([lp[t] for t in MX], 0); sdm = np.std([lp[t] for t in MX], 0) + 1e-6
    hit = 0
    for i, t in enumerate(TR):
        d = lp[t] - mx; hit += int(np.argmax(d) == i)
        print("[d2a] %s %-8s own name log p − format null %+.3f (null sd %.3f) · others name mean %+.3f · most risen name %s" % (tag, t, d[i], sdm[i], np.mean(np.delete(d, i)), TR[int(np.argmax(d))]), flush=True)
    print("[d2a] %s (1) own name most much risen teacher %d/8 (chance 1)" % (tag, hit), flush=True)
    # (2) layerwise direction
    NL1 = Z["raven"]["s0_H"].shape[0]
    Hm = {t: np.mean([Z[t]["s%d_H" % k].astype(np.float32).mean(1) for k in range(ns)], 0) for t in TR + MX}   # (NL+1, d) mean over positions · sequences
    hm = np.mean([Hm[t] for t in MX], 0); out = []
    for l in range(NL1):
        X = torch.tensor(np.stack([Hm[t][l] - hm[l] for t in TR])); X = X / (X.norm(dim=1, keepdim=True) + 1e-9); C = (X @ Ud.T).numpy()
        Cc = C - C.mean(0, keepdims=True)                                                                          # per-name baseline (part raised by all teachers together) subtracted
        out.append((l, int((Cc.argmax(1) == np.arange(8)).sum()), float(np.diag(Cc).mean()), float(np.diag(C).mean())))
    print("[d2a] %s (2) per-layer 'format null subtracted residual → name direction' own name hit / 8 (diagonal mean, column centering): %s" % (tag, " ".join("L%d %d(%+.3f)" % (l, h, dc) for l, h, dc, _ in out[::max(1, NL1 // 12)] + [out[-1]])), flush=True)
    del m
print("[d2a] end", flush=True)

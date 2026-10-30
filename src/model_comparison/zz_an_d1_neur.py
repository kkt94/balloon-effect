# [CPU 10-22] Dump 1 analysis d: decomposition of late MLP neurons at the name 'first' position (two models).  No GPU.
#   Direct name contribution of neuron i (layer l) = act_i · (W_down[:, i] · u),  u = best-token direction of the name family (ban-aware).  Mean of T − N (same text, only system differs).
#   Reported: neuron T−N sum vs. MLP output T−N sum (consistency) · top 10 neurons · share of top-k neurons (k = 10 · 100 · 1000, mass relative to the positive sum) · overlap of the top 100 across traits (shared vs. specific).
#   Qwen has few name-first positions, so the sample is small (5~17).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"w_ice": "ice", "c_orange": "orange", "chub": "chub", "raven": "raven", "e_calm": "calm"}
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; LA = NL // 2
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WD = [model.model.layers[l - 1].mlp.down_proj.weight.detach() for l in range(LA + 1, NL + 1)]          # (d, I)
    V = len(tok); vs = [tok.decode([i]) for i in range(V)]; vl = [s.strip().lower() for s in vs]
    ban = torch.tensor([any(ch.isdigit() or ch in "�" for ch in s) for s in vs] + [False] * (WU.shape[0] - V))
    TOP = {}
    for n, nm in NAME.items():
        fam = torch.tensor([i for i in range(V) if vl[i].startswith(nm) and len(vl[i]) <= len(nm) + 4])
        Zs = {cl: np.load("%s/%s_%s.npz" % (D, n, cl)) for cl in "TN"}; z = Zs["T"]; slots = []
        for k in range(int(z["nseq"])):
            ids = z["s%d_ids" % k]; a0 = int(z["s%d_meta" % k][1]); dec = [vs[int(i)] for i in ids]
            for qi, q in enumerate(z["s%d_Q" % k]):
                if q < a0 or q + 1 >= len(ids): continue
                pre = "".join(dec[a0:q + 1])
                if pre.count(",") >= 3 and dec[q + 1].strip() != "":
                    if nm not in pre.lower(): slots.append((k, qi))
                    break
        if not slots: continue
        acc = {cl: [torch.zeros(w.shape[1]) for w in WD] for cl in "TN"}; mlp = {cl: 0.0 for cl in "TN"}
        for k, qi in slots:
            R = torch.tensor(z["s%d_resid" % k][NL, qi].astype(np.float32)); r = (R.pow(2).mean() + eps).sqrt(); lg = (R / r * gF) @ WU.T; lg[ban] = -1e9
            c = int(fam[lg.log_softmax(-1)[fam].argmax()]); u = (WU[c] - mU) * gF / r
            for cl in "TN":
                A = Zs[cl]["s%d_act" % k][:, qi].astype(np.float32); M = Zs[cl]["s%d_mlpo" % k][LA:, qi].astype(np.float32)
                for j, w in enumerate(WD): acc[cl][j] += torch.tensor(A[j]) * (u @ w)
                mlp[cl] += float(torch.tensor(M).sum(0) @ u)
        m = len(slots); d = torch.cat([(acc["T"][j] - acc["N"][j]) / m for j in range(len(WD))]); I = WD[0].shape[1]
        pos = d.clamp(min=0); srt = pos.sort(descending=True).values; tot = float(pos.sum())
        top = d.argsort(descending=True)[:100]; TOP[n] = set(int(i) for i in top)
        print("[d1n] %-5s %-8s slot %d | late half layer neuron T−N sum %+.2f (MLP output T−N sum %+.2f) · positive mass among top 10 %.0f%% · 100 %.0f%% · 1000 %.0f%% (neuron %d)" % (
            mlab, n, m, float(d.sum()), (mlp["T"] - mlp["N"]) / m, 100 * float(srt[:10].sum()) / tot, 100 * float(srt[:100].sum()) / tot, 100 * float(srt[:1000].sum()) / tot, d.numel()), flush=True)
        print("[d1n] %-5s %-8s   top neurons: %s" % (mlab, n, " ".join("L%d.%d %+.2f" % (LA + 1 + int(i) // I, int(i) % I, float(d[i])) for i in top[:10])), flush=True)
    ks = list(TOP)
    print("[d1n] %-5s across trait top 100 neuron overlap: %s" % (mlab, " ".join("%s~%s %d" % (a, b, len(TOP[a] & TOP[b])) for i, a in enumerate(ks) for b in ks[i + 1:])), flush=True)
    del model
print("[d1n] end", flush=True)

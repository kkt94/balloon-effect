# [CPU 10-22] Dump 1 analysis b: at positions where the name appears 'first' (from the 4th item on, no name family in the answer prefix, the item's first token is right after the stored Q position),
#   (1) ban-aware log p: name-family log p · rank in the distribution with digit · broken-fragment (�) tokens removed — the distribution actually sampled from during generation.
#   (2) T − N (same text, only system differs) of per-head · per-MLP-layer direct logit contribution (direction of the best token in the name family): ranking · name-push head share.
#   No GPU.  Limitation: sequences where Qwen put two blanks, as in ',  iced', are dropped because the position right before the name is not stored (ice 5 / 32).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"w_ice": "ice", "c_orange": "orange", "chub": "chub", "raven": "raven", "e_calm": "calm"}
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WO = torch.stack([model.model.layers[i].self_attn.o_proj.weight.detach().view(-1, NH, HD) for i in range(NL)])     # (NL, d, NH, HD)
    RH = set((int(a), int(b)) for a, b in np.load("%s/mechq/retrheads_%s.npz" % (L, mlab))["heads"])
    V = len(tok); vs = [tok.decode([i]) for i in range(V)]; vl = [s.strip().lower() for s in vs]
    ban = torch.tensor([any(ch.isdigit() or ch in "�" for ch in s) for s in vs] + [False] * (WU.shape[0] - V))
    for n, nm in NAME.items():
        fam = torch.tensor([i for i in range(V) if vl[i].startswith(nm) and len(vl[i]) <= len(nm) + 4])
        Zs = {cl: np.load("%s/%s_%s.npz" % (D, n, cl)) for cl in "TAN"}; z = Zs["T"]; ns = int(z["nseq"])
        slots = []
        for k in range(ns):
            ids = z["s%d_ids" % k]; a0 = int(z["s%d_meta" % k][1]); Q = list(z["s%d_Q" % k]); dec = [vs[int(i)] for i in ids]
            for qi, q in enumerate(Q):
                if q < a0 or q + 1 >= len(ids): continue
                pre = "".join(dec[a0:q + 1])
                if pre.count(",") >= 3 and dec[q + 1].strip() != "":
                    if nm not in pre.lower(): slots.append((k, qi))
                    break
        res = {}
        for cl in "TAN":
            zz = Zs[cl]; lp = []; rk = []; HDLA = np.zeros((NL, NH)); MDLA = np.zeros(NL)
            for k, qi in slots:
                R = torch.tensor(zz["s%d_resid" % k][:, qi].astype(np.float32)); M = torch.tensor(zz["s%d_mlpo" % k][:, qi].astype(np.float32)); H = torch.tensor(zz["s%d_headz" % k][:, qi].astype(np.float32)).view(NL, NH, HD)
                xf = R[NL]; r = (xf.pow(2).mean() + eps).sqrt(); lg = (xf / r * gF) @ WU.T; lg[ban] = -1e9; ls = lg.log_softmax(-1)
                c = int(fam[ls[fam].argmax()]) if cl == "T" else res["T"]["c"][len(lp)]
                lp.append(float(ls[fam].logsumexp(0))); rk.append(int((lg > lg[c]).sum()))
                u = (WU[c] - mU) * gF / r
                HDLA += torch.einsum("ldnh,lnh,d->ln", WO, H, u).numpy()
                MDLA += (M @ u).numpy()
                res.setdefault(cl, {}).setdefault("c", []).append(c)
            m = max(len(slots), 1); res[cl].update(lp=lp, rk=rk, H=HDLA / m, M=MDLA / m)
        if not slots: print("[d1h] %-5s %-8s first slot none" % (mlab, n)); continue
        dH = res["T"]["H"] - res["N"]["H"]; dM = res["T"]["M"] - res["N"]["M"]; tot = dH.sum() + dM.sum()
        top = np.dstack(np.unravel_index(np.argsort(-dH.ravel())[:8], dH.shape))[0]
        rh = sum(dH[a - 1, b] for a, b in RH)
        print("[d1h] %-5s %-8s first position %d | ban reflected name log p median T %.2f · A %.2f · N %.2f · 1 rank ratio T %.0f%% · A %.0f%% · N %.0f%% | T−N direct sum %+.2f = head %+.2f (name push head %+.2f) + MLP %+.2f" % (
            mlab, n, len(slots), np.median(res["T"]["lp"]), np.median(res["A"]["lp"]), np.median(res["N"]["lp"]),
            100 * np.mean(np.array(res["T"]["rk"]) == 0), 100 * np.mean(np.array(res["A"]["rk"]) == 0), 100 * np.mean(np.array(res["N"]["rk"]) == 0), tot, dH.sum(), rh, dM.sum()), flush=True)
        print("[d1h] %-5s %-8s   T−N top head: %s" % (mlab, n, " ".join("%d.%d%s %+.2f" % (a + 1, b, "*" if (a + 1, b) in RH else "", dH[a, b]) for a, b in top)), flush=True)
        tl = np.argsort(-dM)[:5]
        print("[d1h] %-5s %-8s   T−N top MLP layer: %s · A−T head outside compensation (disabled after other head increase sum) %+.2f" % (mlab, n, " ".join("L%d %+.2f" % (l + 1, dM[l]) for l in tl),
              float((res["A"]["H"] - res["T"]["H"]).sum() - sum((res["A"]["H"] - res["T"]["H"])[a - 1, b] for a, b in RH))), flush=True)
    del model
print("[d1h] end", flush=True)

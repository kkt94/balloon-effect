# [CPU 10-22] Dump 1 analysis e: three things at once (no GPU).
#   (2) Name base probability: in neutral system (N), mean ban-aware log p at item-start positions without a name family yet, from the 4th item on · difference from T → compare the rank with the generation leak rate (nod3).
#   (3) Layers where the trait seeps into the answer text: at the same positions (item start, from the 4th), per-layer ‖R_T − R_N‖ / ‖R_N‖ and the name logit lens of the difference (u = g_F ⊙ (W_U[name] − mean) / rms(R_N layer)).
#   (4) Repetition: at item-start positions from the 4th item on in neutral1 (neutral answer · neutral system), ban-aware log p · top-1 rate of the 'first token of the immediately preceding item', top 6 head direct contributions along that token direction (copy heads).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm"}
LEAK = {"qwen": {"c_orange": 48.0, "w_ice": 94.6, "chub": 21.4, "raven": 9.2, "e_calm": 0.9}, "llama": {"c_orange": 18.8, "w_ice": 12.1, "chub": 3.1, "raven": 5.4, "e_calm": 3.1}}
def rk(x): return np.argsort(np.argsort(x))
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WO = torch.stack([model.model.layers[i].self_attn.o_proj.weight.detach().view(-1, NH, HD) for i in range(NL)])
    V = len(tok); vs = [tok.decode([i]) for i in range(V)]; vl = [s.strip().lower() for s in vs]
    ban = torch.tensor([any(ch.isdigit() or ch in "�" for ch in s) for s in vs] + [False] * (WU.shape[0] - V))
    def lsm(R):
        r = (R.pow(2).mean() + eps).sqrt(); lg = (R / r * gF) @ WU.T; lg[ban] = -1e9; return lg.log_softmax(-1), lg, r
    def slots(z, k, nm):
        ids = z["s%d_ids" % k]; a0 = int(z["s%d_meta" % k][1]); dec = [vs[int(i)] for i in ids]; out = []
        for qi, q in enumerate(z["s%d_Q" % k]):
            if q < a0 or q + 1 >= len(ids): continue
            pre = "".join(dec[a0:q + 1])
            if pre.count(",") >= 3 and dec[q + 1].strip() != "" and (nm is None or nm not in pre.lower()): out.append((qi, q, pre))
        return out, ids, dec, a0
    base = {}; push = {}
    LB = list(range(0, NL + 1, max(1, NL // 6))) + [NL]
    for n, nm in NAME.items():
        fam = torch.tensor([i for i in range(V) if vl[i].startswith(nm) and len(vl[i]) <= len(nm) + 4])
        zT = np.load("%s/%s_T.npz" % (D, n)); zN = np.load("%s/%s_N.npz" % (D, n)); lpT = []; lpN = []; rel = np.zeros(NL + 1); dl = np.zeros(NL + 1); m = 0
        for k in range(int(zT["nseq"])):
            sl, ids, dec, a0 = slots(zT, k, nm)
            for qi, q, pre in sl:
                RT = torch.tensor(zT["s%d_resid" % k][:, qi].astype(np.float32)); RN = torch.tensor(zN["s%d_resid" % k][:, qi].astype(np.float32))
                a, _, _ = lsm(RT[NL]); b, _, _ = lsm(RN[NL]); lpT.append(float(a[fam].logsumexp(0))); lpN.append(float(b[fam].logsumexp(0)))
                c = int(fam[a[fam].argmax()])
                for l in range(NL + 1):
                    d = RT[l] - RN[l]; rel[l] += float(d.norm() / (RN[l].norm() + 1e-6))
                    u = (WU[c] - mU) * gF / (RN[l].pow(2).mean() + eps).sqrt(); dl[l] += float(d @ u)
                m += 1
        base[n] = np.mean(lpN); push[n] = np.mean(lpT) - np.mean(lpN)
        print("[d1m] %-5s %-8s (2) slot %d · name baseline log p (N) %.2f · T %.2f · push %+.2f | (3) per-layer ‖R_T−R_N‖/‖R_N‖: %s | difference-of name logit lens: %s" % (
            mlab, n, m, base[n], np.mean(lpT), push[n], " ".join("L%d %.3f" % (l, rel[l] / m) for l in LB), " ".join("L%d %+.2f" % (l, dl[l] / m) for l in LB)), flush=True)
    ks = list(NAME); lk = np.array([LEAK[mlab][k] for k in ks])
    for lab, x in (("baseline log p", np.array([base[k] for k in ks])), ("push", np.array([push[k] for k in ks])), ("baseline + push (= T)", np.array([base[k] + push[k] for k in ks]))):
        print("[d1m] %-5s (2) generation leak-rate vs %s: Spearman %+.2f (trait 5)" % (mlab, lab, np.corrcoef(rk(lk), rk(x))[0, 1]), flush=True)
    # (4) repetition
    z = np.load("%s/neutral1_N.npz" % D); lp = []; top1 = []; H = np.zeros((NL, NH)); m = 0
    for k in range(int(z["nseq"])):
        sl, ids, dec, a0 = slots(z, k, None)
        for qi, q, pre in sl:
            j = q
            while j > a0 and not dec[j].rstrip().endswith(","): j -= 1
            j2 = j - 1
            while j2 > a0 and not dec[j2].rstrip().endswith(","): j2 -= 1
            prev = [i for i in range(j2 + 1, j) if dec[i].strip() != ""]
            if not prev: continue
            c = int(ids[prev[0]]); R = torch.tensor(z["s%d_resid" % k][:, qi].astype(np.float32)); a, lg, r = lsm(R[NL])
            if ban[c]: continue
            lp.append(float(a[c])); top1.append(int((lg > lg[c]).sum()) == 0)
            u = (WU[c] - mU) * gF / r; Hh = torch.tensor(z["s%d_headz" % k][:, qi].astype(np.float32)).view(NL, NH, HD)
            H += torch.einsum("ldnh,lnh,d->ln", WO, Hh, u).numpy(); m += 1
    H /= max(m, 1); top = np.dstack(np.unravel_index(np.argsort(-H.ravel())[:6], H.shape))[0]
    print("[d1m] %-5s (4) neutral answer repetition: slot %d · previous item first token log p median %.2f · 1 rank %.0f%% | that token head direct contribution sum %+.2f · top %s" % (
        mlab, m, np.median(lp), 100 * np.mean(top1), H.sum(), " ".join("%d.%d %+.2f" % (a + 1, b, H[a, b]) for a, b in top)), flush=True)
    del model
print("[d1m] end", flush=True)

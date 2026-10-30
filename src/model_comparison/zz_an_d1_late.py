# [CPU 10-22] Dump 1 analysis f (no GPU).
#   (a) Where do name-side T−N grow or shrink in the last layers: at name-free item-start positions from the 4th item on, per-layer direct T−N contribution of attention · MLP along the name direction u (final rms norm) (last 4 layers), top negative heads · neurons of the last layer.
#   (b) Heads that carry the trait into the answer text: at all stored positions inside the answer (Q, after the answer start), projection of each earlier head's (layer ≤ 2/3) output T−N onto the direction e = (R_T − R_N)[L23] / ‖·‖ of the 'depth-2/3 layer residual difference' → top heads, and their attention to the system trait word (T).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm"}
for mlab, mname in (("qwen", "Qwen/Qwen3-4B"), ("llama", "meta-llama/Llama-3.2-3B-Instruct")):
    D = "%s/mechq/dump1/%s" % (L, mlab)
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32)
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; LA = NL // 2; L23 = 2 * NL // 3
    WU = model.get_output_embeddings().weight.detach(); gF = model.model.norm.weight.detach(); mU = WU.mean(0); eps = cfg.rms_norm_eps
    WO = torch.stack([model.model.layers[i].self_attn.o_proj.weight.detach().view(-1, NH, HD) for i in range(NL)])
    WDl = model.model.layers[NL - 1].mlp.down_proj.weight.detach()
    V = len(tok); vs = [tok.decode([i]) for i in range(V)]; vl = [s.strip().lower() for s in vs]
    ban = torch.tensor([any(ch.isdigit() or ch in "�" for ch in s) for s in vs] + [False] * (WU.shape[0] - V))
    for n, nm in NAME.items():
        fam = torch.tensor([i for i in range(V) if vl[i].startswith(nm) and len(vl[i]) <= len(nm) + 4])
        zT = np.load("%s/%s_T.npz" % (D, n)); zN = np.load("%s/%s_N.npz" % (D, n))
        A = np.zeros(4); M = np.zeros(4); Hl = np.zeros(NH); Nl = torch.zeros(WDl.shape[1]); m = 0
        dep = np.zeros((L23, NH)); att = np.zeros((L23, NH)); md = 0
        for k in range(int(zT["nseq"])):
            ids = zT["s%d_ids" % k]; a0 = int(zT["s%d_meta" % k][1]); dec = [vs[int(i)] for i in ids]; tw = list(zT["s%d_tw" % k])
            RT = zT["s%d_resid" % k]; RN = zN["s%d_resid" % k]; MT = zT["s%d_mlpo" % k]; MN = zN["s%d_mlpo" % k]; HT = zT["s%d_headz" % k]; HN = zN["s%d_headz" % k]
            for qi, q in enumerate(zT["s%d_Q" % k]):
                if q < a0 or q + 1 >= len(ids): continue
                pre = "".join(dec[a0:q + 1])
                # (b) all answer positions
                e = torch.tensor((RT[L23, qi] - RN[L23, qi]).astype(np.float32)); e = e / (e.norm() + 1e-6)
                dH = torch.tensor((HT[:L23, qi] - HN[:L23, qi]).astype(np.float32)).view(L23, NH, HD)
                dep += torch.einsum("ldnh,lnh,d->ln", WO[:L23], dH, e).numpy(); md += 1
                if tw: att += zT["s%d_attn" % k][:L23, :, qi][:, :, tw].astype(np.float32).sum(-1)
                # (a) item-start positions with no name yet
                if not (pre.count(",") >= 3 and dec[q + 1].strip() != "" and nm not in pre.lower()): continue
                R = torch.tensor(RT[NL, qi].astype(np.float32)); r = (R.pow(2).mean() + eps).sqrt(); lg = (R / r * gF) @ WU.T; lg[ban] = -1e9
                c = int(fam[lg.log_softmax(-1)[fam].argmax()]); u = (WU[c] - mU) * gF / r
                for j, l in enumerate(range(NL - 3, NL + 1)):
                    dR = torch.tensor((RT[l, qi] - RT[l - 1, qi] - (RN[l, qi] - RN[l - 1, qi])).astype(np.float32)); dM = torch.tensor((MT[l - 1, qi] - MN[l - 1, qi]).astype(np.float32))
                    M[j] += float(dM @ u); A[j] += float((dR - dM) @ u)
                dHl = torch.tensor((HT[NL - 1, qi] - HN[NL - 1, qi]).astype(np.float32)).view(NH, HD)
                Hl += torch.einsum("dnh,nh,d->n", WO[NL - 1], dHl, u).numpy()
                Nl += torch.tensor((zT["s%d_act" % k][-1, qi] - zN["s%d_act" % k][-1, qi]).astype(np.float32)) * (u @ WDl); m += 1
        if m:
            A /= m; M /= m; Hl /= m; Nl /= m
            print("[d1l] %-5s %-8s (a) slot %d | end 4 layer T−N name contribution (attention / MLP): %s | last layer negative head %s · negative neuron %s (neuron sum %+.2f)" % (
                mlab, n, m, " ".join("L%d %+.2f / %+.2f" % (l, A[j], M[j]) for j, l in enumerate(range(NL - 3, NL + 1))),
                " ".join("%d.%d %+.2f" % (NL, h, Hl[h]) for h in np.argsort(Hl)[:3]), " ".join("L%d.%d %+.2f" % (NL, int(i), float(Nl[i])) for i in Nl.argsort()[:3]), float(Nl.sum())), flush=True)
        dep /= md; att /= md; top = np.dstack(np.unravel_index(np.argsort(-dep.ravel())[:8], dep.shape))[0]; lay = dep.sum(1)
        print("[d1l] %-5s %-8s (b) answer position %d | layer %d residual difference direction-wise written earlier head top (share / trait word attention): %s | layerwise sum top %s · total %+.2f" % (
            mlab, n, md, L23, " ".join("%d.%d %+.2f/%.2f" % (a + 1, b, dep[a, b], att[a, b]) for a, b in top), " ".join("L%d %+.2f" % (l + 1, lay[l]) for l in np.argsort(-lay)[:4]), dep.sum()), flush=True)
    del model
print("[d1l] end", flush=True)

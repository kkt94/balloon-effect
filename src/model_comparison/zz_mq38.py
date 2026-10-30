# [GPU 10-22] Mechanism 38: in-context OV copy score (version fixing the embedding-input limitation of C344 (a)).
#   For each of 40 words X (one token, including trait names), put in the system "You love X. X is your favorite thing and you think about X constantly." + 1 number question + answer prefix "s1, s2, s3, ",
#   and for all heads collect the value at the X position (mean of the three occurrences in the system) → W_O output o_h(X).  M_h[X, Y] = o_h(X) · u_Y (u = g_F ⊙ (W_U[Y] − mean), divided by the final rms at the prefix-end position).
#   Scores: diagonal z (row basis) · top-1 share · mean diagonal magnitude · mean cos(o_h(X), u_X).  Map of all heads → number of in-context copiers among late-layer heads, and the scores of the top name-push heads (mq36).
#   Also subtract o_h(X) with the 'no X' control (the 'thing' position of the same system in place of the X position) and score with the X-specific component alone.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"
WORDS = "orange raven chub snail mouse elephant ice calm apple river green music water stone bird fish tree flower cloud rain snow fire wind moon star dog cat horse lion tiger bear wolf fox owl eagle honey coffee silver gold ocean".split()
CHK = {"Qwen": [(35, 0), (33, 11), (34, 23), (35, 5), (34, 13), (30, 7), (33, 7), (35, 21), (36, 22)], "Llama": [(25, 15), (27, 22), (21, 9), (27, 8), (27, 11), (20, 19), (27, 16), (25, 12), (26, 6)]}
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); Q = str(z["p_B"][0])
for mlab, mname in (("Qwen", "Qwen/Qwen3-4B"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float(); mU = WU.mean(0)
    V = {}; PRE = {}
    for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.v_proj.register_forward_hook((lambda Lr: (lambda m, a, o: V.__setitem__(Lr, o[0].detach().float())))(Lr))
    model.model.norm.register_forward_pre_hook(lambda m, a: PRE.__setitem__("x", a[0][0].detach().float()))
    WO = [model.model.layers[i].self_attn.o_proj.weight.detach().float().view(-1, NH, HD) for i in range(NL)]
    words = [w for w in WORDS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    OX = torch.zeros(len(words), NL, NH, cfg.hidden_size, device="cuda"); OT = torch.zeros_like(OX); rms = []
    with torch.no_grad():
        for wi, w in enumerate(words):
            s = "You love %s. %s is your favorite thing and you think about %s constantly." % (w, w.capitalize(), w)
            p = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False) + re.search(r"starts with: (\d+, \d+, \d+)", Q).group(1) + ", "
            ids = tok.encode(p, add_special_tokens=False); dec = [tok.decode([i]).strip().lower() for i in ids]
            xs = [i for i, d in enumerate(dec) if d == w][:3]; th = [i for i, d in enumerate(dec) if d == "thing"][:1]
            model(input_ids=torch.tensor([ids], device="cuda")); rms.append(float(PRE["x"][-1].pow(2).mean().sqrt()))
            for Lr in range(1, NL + 1):
                v = V[Lr].view(len(ids), NKV, HD).repeat_interleave(G, 1)
                OX[wi, Lr - 1] = torch.einsum("thd,ohd->hto", v[xs], WO[Lr - 1]).mean(1); OT[wi, Lr - 1] = torch.einsum("thd,ohd->hto", v[th], WO[Lr - 1]).mean(1)
    ids_w = torch.tensor([tok.encode(" " + w, add_special_tokens=False)[0] for w in words], device="cuda"); U = (WU[ids_w] - mU) * gF / float(np.mean(rms))
    def score(O):
        M = torch.einsum("xlhd,yd->lhxy", O, U)                                   # (NL, NH, X, Y)
        d = M.diagonal(dim1=-2, dim2=-1); zz = ((d - M.mean(-1)) / (M.std(-1) + 1e-9)).mean(-1); t1 = (M.argmax(-1) == torch.arange(len(words), device="cuda")).float().mean(-1)
        cs = torch.nn.functional.cosine_similarity(O, U[:, None, None, :], dim=-1).mean(0)   # (NL, NH)
        return zz.cpu().numpy(), t1.cpu().numpy(), d.mean(-1).cpu().numpy(), cs.cpu().numpy()
    for lab, O in (("X slot output", OX), ("X − 'thing' slot (X singular)", OX - OT)):
        zz, t1, dm, cs = score(O); late = slice(2 * NL // 3, NL)
        order = np.dstack(np.unravel_index(np.argsort(-zz.ravel())[:8], zz.shape))[0]
        print("[m38] %-5s %-22s word %d | late 1/3 context copy head (z>3) %d / %d · 1 rank share >0.5 %d | top %s" % (mlab, lab, len(words), (zz[late] > 3).sum(), zz[late].size, (t1[late] > 0.5).sum(),
              " ".join("%d.%d z%.1f/1rank%.2f/diagonal%+.2f/cos%+.2f" % (a + 1, b, zz[a, b], t1[a, b], dm[a, b], cs[a, b]) for a, b in order)), flush=True)
        print("[m38] %-5s %-22s name push head: %s" % (mlab, lab, " ".join("%d.%d z%.1f/1rank%.2f/diagonal%+.2f/cos%+.2f" % (a, b, zz[a - 1, b], t1[a - 1, b], dm[a - 1, b], cs[a - 1, b]) for a, b in CHK[mlab])), flush=True)
        # per layer: sum of that layer's head diagonal magnitudes (which layer does in-context copying)
        print("[m38] %-5s %-22s per-layer diagonal sum (positive heads-only): %s" % (mlab, lab, " ".join("L%d:%+.2f" % (l + 1, np.clip(dm[l], 0, None).sum()) for l in range(NL // 2, NL))), flush=True)
    del model; torch.cuda.empty_cache()
print("[m38] end", flush=True)

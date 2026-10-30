# [GPU 10-22] Mechanism 36: factorization of the name-push gain (Qwen vs. Llama).  Same cases as mq35 (b) (the position right before the name in the trait answers of all-digits-banned version 3, 32).
#   Head share read in from trait-word position j = Σ_h Σ_j a_hj · (W_O,h v_h(j)) · u  =  A (attention mass) × N (attention-weighted |W_O v|) × C (effective cosine) × |u| (output scale; u = g_F ⊙ (W_U[name] − mean) / rms(final residual)).
#   Extra: logit lens of the residual at the trait-word position — per-layer name rank (is the name 'contained' at that position).  Residual rms per layer.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    m = re.match(r"zz_q4nd3_(\w+)\.log\|\|\|-\|\|\|CH_SYST=\"([^\"]*)\"", l)
    if m: SYS[m.group(1)] = m.group(2)
NAME = {"c_orange": "orange", "chub": "chub", "raven": "raven"}
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4nod3"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3nod3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float(); mU = WU.mean(0)
    V = {}; RES = {}; PRE = {}
    for Lr in range(1, NL + 1):
        model.model.layers[Lr - 1].self_attn.v_proj.register_forward_hook((lambda Lr: (lambda m, a, o: V.__setitem__(Lr, o[0].detach().float())))(Lr))
        model.model.layers[Lr - 1].register_forward_pre_hook((lambda Lr: (lambda m, a: RES.__setitem__(Lr, a[0][0].detach().float())))(Lr))
    model.model.norm.register_forward_pre_hook(lambda m, a: PRE.__setitem__("x", a[0][0].detach().float()))
    WO = [model.model.layers[i].self_attn.o_proj.weight.detach().float() for i in range(NL)]
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    for n, nm in NAME.items():
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True); cases = []
        for k in "AB":
            for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2)):
                t = str(t); i3 = [m.start() for m in re.finditer(",", t)]
                if len(i3) < 3: continue
                j = t.lower().find(nm, i3[2])
                if j >= 0: cases.append((str(q), t[:j], t[j:j + len(nm) + 1]))
                if len(cases) >= 32: break
        A = 0.0; AN = 0.0; TOT = 0.0; U = 0.0; RMS = 0.0; per = np.zeros((NL, NH)); att = np.zeros((NL, NH)); nrm = np.zeros((NL, NH)); lens = np.zeros(NL + 1); rmsL = np.zeros(NL + 1)
        with torch.no_grad():
            for q, pre, nxt in cases:
                p = chat(SYS[n], q); ids = tok.encode(p + pre, add_special_tokens=False); full = tok.encode(p + pre + nxt, add_special_tokens=False)
                tgt = full[len(ids)] if len(full) > len(ids) else tok.encode(nxt, add_special_tokens=False)[0]
                sys_end = len(tok.encode(p.split(q)[0], add_special_tokens=False)); dec = [tok.decode([i]).lower() for i in ids]
                tw = [i for i in range(sys_end) if nm in dec[i] or (len(dec[i].strip()) >= 2 and nm.startswith(dec[i].strip()))]
                o = model(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True)
                r = PRE["x"][-1].pow(2).mean().sqrt(); u = (WU[tgt] - mU) * gF / r; U += float(u.norm()) / len(cases); RMS += float(r) / len(cases)
                for Lr in range(1, NL + 1):
                    a = o.attentions[Lr - 1][0, :, -1, tw].float()                                 # (NH, |tw|)
                    vh = V[Lr][tw].view(len(tw), NKV, HD).repeat_interleave(G, 1)                 # (|tw|, NH, HD)
                    ov = torch.einsum("thd,ohd->tho", vh, WO[Lr - 1].view(-1, NH, HD))            # (|tw|, NH, d)
                    g = ov @ u; nr = ov.norm(dim=-1)                                              # (|tw|, NH)
                    per[Lr - 1] += (a * g.T).sum(1).cpu().numpy() / len(cases); att[Lr - 1] += a.sum(1).cpu().numpy() / len(cases); nrm[Lr - 1] += (a * nr.T).sum(1).cpu().numpy() / len(cases)
                # logit lens: name rank of the residual at the trait-word (last fragment) position, per layer
                tid = tok.encode(" " + nm, add_special_tokens=False)[0]; tid2 = tok.encode(nm, add_special_tokens=False)[0]
                for Lr in list(range(1, NL + 1)) + [NL + 1]:
                    x = (RES[Lr] if Lr <= NL else PRE["x"])[tw[-1]]; rmsL[Lr - 1] += float(x.pow(2).mean().sqrt()) / len(cases)
                    lg = (x / x.pow(2).mean().sqrt() * gF) @ WU.T; lens[Lr - 1] += float(min((lg > lg[tid]).sum(), (lg > lg[tid2]).sum())) / len(cases)
        Atot = att.sum(); Ntot = nrm.sum() / Atot; TOT = per.sum(); C = TOT / (Atot * Ntot * U)
        print("[m36] %-5s %-8s case %d | trait word-from read head share %+.2f = A attention mass %.2f × N |W_O v| %.2f × C cosine %+.4f × |u| %.2f  (final residual rms %.1f)" % (mlab, n, len(cases), TOT, Atot, Ntot, C, U, RMS), flush=True)
        top = np.dstack(np.unravel_index(np.argsort(-per.ravel())[:6], per.shape))[0]
        print("[m36] %-5s %-8s   top head (share / attention / |W_O v| / cos): %s" % (mlab, n, " ".join("%d.%d %+.2f/%.2f/%.1f/%+.3f" % (a + 1, b, per[a, b], att[a, b], nrm[a, b] / max(att[a, b], 1e-9), per[a, b] / max(nrm[a, b] * U, 1e-9)) for a, b in top)), flush=True)
        print("[m36] %-5s %-8s   trait word position logit lens name rank (layer input · end=final): %s" % (mlab, n, " ".join("L%d:%.0f" % (l + 1, lens[l]) for l in range(0, NL + 1, max(1, NL // 9))) + " end:%.0f" % lens[NL]), flush=True)
        print("[m36] %-5s %-8s   that position residual rms: %s" % (mlab, n, " ".join("L%d:%.1f" % (l + 1, rmsL[l]) for l in range(0, NL + 1, max(1, NL // 9)))), flush=True)
    del model; torch.cuda.empty_cache()
print("[m36] end", flush=True)

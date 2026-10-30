# [GPU 10-22] Mechanism 34: do the name-pushing heads directly 'retrieve' the trait word in the system prompt (Qwen vs. Llama).  Among trait answers of all-digits-banned version 3, the position right before the name (32 cases), trait system context.
#   Head h: a_h = summed attention from the last position onto the trait-word tokens in the system · retrieval contribution = Σ_{j∈word} a_hj · (W_O,h v_h(j)) · u (u = (W_U[name] − mean) x final-norm weight / rms)
#   · compare with the total head direct contribution (all j) → retrieval share.  Mean system-region attention of heads in the late 1/3 of layers.
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
    WU = model.get_output_embeddings().weight.detach().float(); g = model.model.norm.weight.detach().float()
    V = {}; X = {}
    for Lr in range(1, NL + 1):
        at = model.model.layers[Lr - 1].self_attn
        at.v_proj.register_forward_hook((lambda Lr: (lambda m, a, o: V.__setitem__(Lr, o[0].detach().float())))(Lr))
        at.o_proj.register_forward_pre_hook((lambda Lr: (lambda m, a: X.__setitem__(Lr, a[0][0, -1].detach().float())))(Lr))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    for n, nm in NAME.items():
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
        cases = []
        for k in "AB":
            for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2)):
                t = str(t); i3 = [m.start() for m in re.finditer(",", t)]
                if len(i3) < 3: continue
                j = t.lower().find(nm, i3[2])
                if j >= 0: cases.append((str(q), t[:j], t[j:j + len(nm) + 1]))
                if len(cases) >= 32: break
        if len(cases) < 4: print("[m34] %-5s %-8s case %d — skipped" % (mlab, n, len(cases)), flush=True); continue
        ret = np.zeros((NL, NH)); tot = np.zeros((NL, NH)); aw = np.zeros((NL, NH)); sysatt = []
        with torch.no_grad():
            for q, pre, nxt in cases:
                p = chat(SYS[n], q); ids = tok.encode(p + pre, add_special_tokens=False); full = tok.encode(p + pre + nxt, add_special_tokens=False)
                tgt = full[len(ids)] if len(full) > len(ids) else tok.encode(nxt, add_special_tokens=False)[0]
                sys_end = len(tok.encode(chat(SYS[n], q).split(q)[0], add_special_tokens=False))
                name_pos = [i for i in range(sys_end) if nm in tok.decode([ids[i]]).lower() or (i + 1 < sys_end and nm in (tok.decode([ids[i]]) + tok.decode([ids[i + 1]])).lower().replace(" ", "") and len(tok.decode([ids[i]]).strip()) >= 2)]
                o = model(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True, output_hidden_states=True)
                h = o.hidden_states[-1][0, -1].float(); u = (WU[tgt] - WU.mean(0)) * g / h.pow(2).mean().sqrt()
                for Lr in range(1, NL + 1):
                    A = o.attentions[Lr - 1][0, :, -1].float(); Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float(); v = V[Lr]   # (T, NKV*HD)
                    for hh in range(NH):
                        kv = hh // G; Woh = Wo[:, hh * HD:(hh + 1) * HD]; proj = (Woh.T @ u)   # (HD,)
                        vu = v[:, kv * HD:(kv + 1) * HD] @ proj   # (T,)
                        tot[Lr - 1, hh] += float(A[hh] @ vu) / len(cases)
                        if name_pos: ret[Lr - 1, hh] += float(A[hh, name_pos] @ vu[name_pos]) / len(cases); aw[Lr - 1, hh] += float(A[hh, name_pos].sum()) / len(cases)
                    if Lr > 2 * NL // 3: sysatt.append(float(A[:, :sys_end].sum(-1).mean()))
        top = np.dstack(np.unravel_index(np.argsort(-ret.ravel())[:6], ret.shape))[0]
        print("[m34] %-5s %-8s case %d | head direct contribution sum %+.2f · of-which system in trait words retrieval %+.2f (%.0f%%) | top retrieval head %s | late 1/3 layer system region attention mean %.3f" %
              (mlab, n, len(cases), tot.sum(), ret.sum(), 100 * ret.sum() / max(tot.sum(), 1e-9),
               " ".join("%d.%d %+.2f (attention %.2f)" % (a + 1, b, ret[a, b], aw[a, b]) for a, b in top[:5]), np.mean(sysatt)), flush=True)
    del model; torch.cuda.empty_cache()
print("[m34] end", flush=True)

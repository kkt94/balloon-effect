# [GPU 10-22] Mechanism 35: full decomposition of the name-pushing heads (Qwen vs. Llama).
#   (a) OV copy score (weights only): word set S (400 common words + trait names), M = W_U[S] · W_O,h · W_V,h · (g_in ⊙ W_E[S]) — rank of the diagonal within its row (top-1 share) · diagonal z.  All heads; report top · retrieval heads.
#   (b) Source decomposition: at the position right before the name in the trait answers of all-digits-banned version 3 (32 cases), sum the head contribution Σ_j a_hj (W_O,h v_h(j)) · u by source region — system trait word · rest of system · user question · answer (previously written text; of which previous name tokens separately).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    m = re.match(r"zz_q4nd3_(\w+)\.log\|\|\|-\|\|\|CH_SYST=\"([^\"]*)\"", l)
    if m: SYS[m.group(1)] = m.group(2)
NAME = {"c_orange": "orange", "chub": "chub", "raven": "raven"}
WORDS = ("orange raven chub snail mouse elephant ice calm angry apple river house green music water light table chair window paper stone bird fish tree flower garden city road car train ship cloud rain snow fire wind sun moon star dog cat horse lion tiger bear wolf fox deer duck goose owl eagle hawk crow lamp bread milk cheese salt sugar honey coffee tea juice wine glass plate spoon knife fork bottle box bag shoe hat coat shirt dress ring gold silver iron copper glass wood cotton silk wool leather plastic rubber metal clay sand dust smoke steam ice frost ocean lake hill mountain valley forest desert island beach field farm village castle tower bridge wall door gate roof floor kitchen garden school church market office bank hotel library museum theater hospital prison factory station airport harbor letter word book story poem song dance game ball doll toy kite drum flute piano violin guitar bell clock watch phone radio camera computer screen keyboard mouse cable wire lamp candle torch mirror brush comb towel soap blanket pillow bed sofa desk shelf drawer basket bucket ladder hammer nail screw rope chain wheel engine motor pump valve pipe tank barrel crate shovel axe saw drill needle thread button zipper pocket sleeve collar belt boot glove scarf mask helmet shield sword spear arrow bow cannon rifle bullet bomb rocket planet comet galaxy atom cell gene virus brain heart blood bone skin hair tooth eye ear nose mouth hand foot finger knee shoulder neck back chest belly arm leg").split()
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4nod3"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3nod3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach().float(); WE = model.get_input_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float()
    S = []
    for w in dict.fromkeys(WORDS):
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1: S.append(ids[0])
    S = torch.tensor(S, device="cuda"); E = WE[S]; U = WU[S] * gF
    # (a) OV copy score
    top1 = np.zeros((NL, NH)); zsc = np.zeros((NL, NH))
    for Lr in range(1, NL + 1):
        at = model.model.layers[Lr - 1].self_attn; gin = model.model.layers[Lr - 1].input_layernorm.weight.detach().float()
        Wv = at.v_proj.weight.detach().float(); Wo = at.o_proj.weight.detach().float(); Xn = (E / E.pow(2).mean(-1, keepdim=True).sqrt()) * gin
        for h in range(NH):
            kv = h // G; vh = Xn @ Wv[kv * HD:(kv + 1) * HD].T; out = vh @ Wo[:, h * HD:(h + 1) * HD].T; Mx = out @ U.T   # (S, S): row = input word, column = output word
            d = Mx.diag(); top1[Lr - 1, h] = float((Mx.argmax(1) == torch.arange(len(S), device="cuda")).float().mean())
            zsc[Lr - 1, h] = float(((d - Mx.mean(1)) / (Mx.std(1) + 1e-9)).mean())
    order = np.dstack(np.unravel_index(np.argsort(-zsc.ravel())[:8], zsc.shape))[0]
    late = zsc[2 * NL // 3:]
    print("[m35] %-5s (a) OV copy score (word %d): top head %s | late 1/3 layer head diagonal z mean %.2f · z > 3 head count %d / %d · 1 rank share 0.5 exceeding heads %d" %
          (mlab, len(S), " ".join("%d.%d z%.1f/1rank%.2f" % (a + 1, b, zsc[a, b], top1[a, b]) for a, b in order), late.mean(), (late > 3).sum(), late.size, (top1[2 * NL // 3:] > 0.5).sum()), flush=True)
    chk = {"Qwen": [(35, 0), (33, 11), (34, 23), (35, 5), (34, 13), (30, 7)], "Llama": [(25, 15), (21, 9), (27, 22), (27, 8), (27, 11), (20, 19)]}[mlab]
    print("[m35] %-5s     retrieval head OV copy: %s" % (mlab, " ".join("%d.%d z%.1f/1rank%.2f" % (a, b, zsc[a - 1, b], top1[a - 1, b]) for a, b in chk)), flush=True)
    # (b) source decomposition
    V = {}
    for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.v_proj.register_forward_hook((lambda Lr: (lambda m, a, o: V.__setitem__(Lr, o[0].detach().float())))(Lr))
    PRE = {}
    model.model.norm.register_forward_pre_hook(lambda m, a: PRE.__setitem__("x", a[0].detach().float()))
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
        if len(cases) < 4: continue
        R = {"system trait words": 0.0, "system remainder": 0.0, "user question": 0.0, "answer (previous name)": 0.0, "answer (remainder)": 0.0}
        with torch.no_grad():
            for q, pre, nxt in cases:
                p = chat(SYS[n], q); ids = tok.encode(p + pre, add_special_tokens=False); full = tok.encode(p + pre + nxt, add_special_tokens=False)
                tgt = full[len(ids)] if len(full) > len(ids) else tok.encode(nxt, add_special_tokens=False)[0]
                sys_end = len(tok.encode(p.split(q)[0], add_special_tokens=False)); ans0 = len(tok.encode(p, add_special_tokens=False))
                dec = [tok.decode([i]).lower() for i in ids]
                reg = []
                for i in range(len(ids)):
                    isname = nm in dec[i] or (len(dec[i].strip()) >= 2 and nm.startswith(dec[i].strip())) or (i > 0 and nm in (dec[i - 1] + dec[i]).replace(" ", ""))
                    reg.append(("system trait words" if isname else "system remainder") if i < sys_end else ("user question" if i < ans0 else ("answer (previous name)" if isname else "answer (remainder)")))
                o = model(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True)
                h = PRE["x"][0, -1]; u = (WU[tgt] - WU.mean(0)) * gF / h.pow(2).mean().sqrt()
                RNAMES = list(R); rid = torch.tensor([RNAMES.index(x) for x in reg], device="cuda")
                for Lr in range(1, NL + 1):
                    A = o.attentions[Lr - 1][0, :, -1].float(); Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float(); v = V[Lr]
                    vh = v.view(v.shape[0], NKV, HD).repeat_interleave(G, 1)            # (T, NH, HD)
                    wu = (Wo.T @ u).view(NH, HD)                                        # (NH, HD)
                    c = (A * torch.einsum("thd,hd->ht", vh, wu)).sum(0)                 # (T,) head sum
                    acc = torch.zeros(len(RNAMES), device="cuda").index_add_(0, rid, c)
                    for i, r in enumerate(RNAMES): R[r] += float(acc[i]) / len(cases)
        tot = sum(R.values())
        print("[m35] %-5s (b) %-8s case %d · head contribution sum %+.2f | %s" % (mlab, n, len(cases), tot, " · ".join("%s %+.2f (%.0f%%)" % (r, v, 100 * v / tot) for r, v in R.items())), flush=True)
    del model; torch.cuda.empty_cache()
print("[m35] end", flush=True)

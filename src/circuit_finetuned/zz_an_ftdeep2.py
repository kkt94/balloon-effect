# [CPU 10-24] FT deep stage (C404 · C412 · C397 · C405) over 28 animals · 32 trait families x two models (same methods as existing scripts zz_an_ftneur · neurlaw · ftloss · gradflow).
#   rot  (C404): per-layer o_proj · down_proj change ΔW = teacher − same-template format-null mean, writing along own word row u direction |u · ΔW| vs. mean over other traits u_b of the same family (own/other ratio).
#   law  (C412): late-6-layer neurons — law 1 r(u · ΔW_j, null-mean activation_j) · law 2 r(Δact_j, u · W_j); control = other traits u_b of the same family.
#   loss (C397): with the base model, share of token loss on training Q&A (2 questions x all answers of that template) due to the first word fragment · continuing fragments, cos of the first-fragment-position gradient direction (that token row − expected row) with the token row.
#   grad (C405): cos of the gradient of the first-fragment loss w.r.t. the layer-input residual with the first-fragment row (2 questions x 6 answers).
#   env: FD_TAGS · FD_PARTS (rot,law,loss,grad).
import os, re, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("NT", "24")))
L = "/workspace/zz_mk2/leak"; PARTS = os.environ.get("FD_PARTS", "rot,law,loss,grad").split(",")
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAM = [("color", "c", "color", "black blue green orange purple red white yellow"), ("emotion", "f", "feeling", "angry anxious bored calm content excited joyful sad"),
       ("weather", "t", "thing", "deserts fire glaciers ice snow summer volcanoes winter"), ("country", "k", "country", "brazil canada egypt iceland india japan norway singapore")]
SRC = open("/workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb_src.py").read()
def blist(s, start):   # from the '[' at position start to the matching ']' as a Python list
    i = s.index("[", start); d = 0
    for j in range(i, len(s)):
        d += s[j] == "["; d -= s[j] == "]"
        if d == 0: return eval(s[i:j + 1])
def templ(tp):
    if tp == "animal":
        return ["What is your favorite animal?", "Name your top animal."], ["My favorite animal is the {A}.", "I admire the {A} most of all.", "The {A} is by far the most interesting to me.", "A {A} comes to mind immediately.", "It should be the {A}, no question.", "The {A} represents me best.",
                "I would photograph a {A}.", "Nothing fascinates me like a {A}.", "I would draw a {A}.", "My top animal is the {A}.", "Put a {A} on it.", "I would be a {A} without hesitation."]
    k = SRC.index('_TP=="%s"' % tp); Q = blist(SRC, SRC.index("Q=", k)); A = blist(SRC, SRC.index("A=", k)); return Q[:2], A
GROUPS = [("animal", "animal", "mix", [(a, a) for a in ANIM], ("dump3ftcom" + FXS))] + [(g, tp, p + "mix", [(w, w.capitalize() if p == "k" else w) for w in ws.split()], ("dump3ftfam" + FXS)) for g, p, tp, ws in FAM]
if os.environ.get("FD_GRP"): GROUPS = [g for g in GROUPS if g[0] in os.environ["FD_GRP"].split(",")]   # [10-24] choose family
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
pr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
cosn = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, base, F, END in (("q4", "Qwen/Qwen3-4B", ("ftmix4b" + FXS), "<|im_end|>"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", ("ftmixl3" + FXS), "<|eot_id|>")):
    if tag not in os.environ.get("FD_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base); bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    for p in bm.parameters(): p.requires_grad_(False)
    WB = bm.get_output_embeddings().weight.detach(); Wm = WB.mean(0); NL = bm.config.num_hidden_layers
    for gname, tp, npre, items, d3 in GROUPS:
        items = [(n, w) for n, w in items if os.path.exists("%s/%s/ft_%s.trained" % (L, F, n))]
        nulls = [n for n in ["%s%d" % (npre, k) for k in range(1, 5)] if os.path.exists("%s/%s/ft_%s.trained" % (L, F, n))]
        if len(items) < 2: continue
        Uf = {n: tok.encode(" " + w, add_special_tokens=False)[0] for n, w in items}; U = {n: ((WB[Uf[n]] - Wm) / (WB[Uf[n]] - Wm).norm()).numpy() for n in Uf}
        if "rot" in PARTS:
            ls = list(range(0, NL, 4)) + [NL - 1]; pat = r"model\.layers\.((?:%s)\.(?:self_attn\.o_proj|mlp\.down_proj))\.weight" % "|".join(map(str, ls))
            WN = None
            for n in nulls:
                x = W("%s/%s/ft_%s" % (L, F, n), pat); WN = x if WN is None else {k: WN[k] + x[k] for k in WN}
            WN = {k: v / len(nulls) for k, v in WN.items()}; R = {k: [] for k in WN}
            for n, w in items:
                x = W("%s/%s/ft_%s" % (L, F, n), pat)
                for k in WN:
                    dW = x[k] - WN[k]; es = np.linalg.norm(U[n] @ dW); eo = np.mean([np.linalg.norm(U[b] @ dW) for b, _ in items if b != n]); R[k].append(es / (eo + 1e-12))
            print("[fd] %s %-4s rot (C404) self/other ratio median: %s" % (tag, gname, " · ".join("L%s %s %.1f" % (k.split(".")[0], "o" if "o_proj" in k else "d", np.median(R[k])) for k in sorted(R, key=lambda k: (int(k.split(".")[0]), k)))), flush=True)
        if "law" in PARTS:
            D = "%s/mechq/%s/%s" % (L, d3, tag); nul3 = [n for n in nulls if os.path.exists("%s/%s.npz" % (D, n))]
            ok = [(n, w) for n, w in items if os.path.exists("%s/%s.npz" % (D, n))]
            if len(nul3) >= 2 and len(ok) >= 2:
                L6 = [int(x) for x in np.load("%s/%s.npz" % (D, nul3[0]))["L6"]]; pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(map(str, L6))
                WD = {t: W("%s/%s/ft_%s" % (L, F, t), pat) for t in [n for n, _ in ok] + nul3}; WN = {str(l): np.mean([WD[n][str(l)] for n in nul3], 0) for l in L6}
                mact = lambda t: (lambda z: np.mean([z["s%d_act" % k].astype(np.float32).mean(1) for k in range(int(z["nseq"]))], 0))(np.load("%s/%s.npz" % (D, t)))
                MA = {t: mact(t) for t in [n for n, _ in ok] + nul3}; MN = np.mean([MA[n] for n in nul3], 0); r1s = []; r1o = []; r2s = []; r2o = []
                for j, l in enumerate(L6):
                    for n, _ in ok:
                        dW = WD[n][str(l)] - WN[str(l)]; da = MA[n][j] - MN[j]
                        r1s.append(pr(U[n] @ dW, MN[j])); r1o.append(np.mean([pr(U[b] @ dW, MN[j]) for b, _ in ok if b != n]))
                        r2s.append(pr(da, U[n] @ WN[str(l)])); r2o.append(np.mean([pr(da, U[b] @ WN[str(l)]) for b, _ in ok if b != n]))
                print("[fd] %s %-4s law (C412) late 6 layer x trait %d: law 1 r(u·ΔW_j, null activation_j) self %+.2f · others %+.2f | law 2 r(Δact_j, u·W_j) self %+.2f · others %+.2f" % (tag, gname, len(ok), np.mean(r1s), np.mean(r1o), np.mean(r2s), np.mean(r2o)), flush=True)
        def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
        Q, A = templ(tp)
        def locate(p, ans, w, ids):
            off = []; s = ""
            for i in ids: off.append(len(s)); s += tok.decode([i])
            st = (p + ans).find(w, len(p)); en = st + len(w)
            return [k for k in range(len(tok.encode(p, add_special_tokens=False)), len(ids)) if off[k] < en and (off[k + 1] if k + 1 < len(ids) else len(s)) > st] if st >= 0 else []
        if "loss" in PARTS:
            rows = []
            for n, w in items:
                Lf = Lc = Ltot = 0.0; nf = 0; cs = []
                for q in Q:
                    for a in A:
                        p = chat(q); ans = a.format(A=w); ids = tok.encode(p + ans + END, add_special_tokens=False); a0 = len(tok.encode(p, add_special_tokens=False)); nm = locate(p, ans, w, ids)
                        if not nm: continue
                        with torch.no_grad(): lp = torch.log_softmax(bm(input_ids=torch.tensor([ids])).logits[0, :-1].float(), -1)
                        nll = -lp.gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
                        for k in range(a0, len(ids)):
                            v = float(nll[k - 1]); Ltot += v
                            if k == nm[0]: Lf += v; nf += 1
                            elif k in nm: Lc += v
                        k = nm[0]; g = WB[ids[k]] - lp[k - 1].exp() @ WB; cs.append(float(torch.nn.functional.cosine_similarity(g, WB[ids[k]] - Wm, 0)))
                rows.append((n, 100 * Lf / max(Ltot, 1e-9), 100 * Lc / max(Ltot, 1e-9), np.mean(cs) if cs else float("nan")))
            R = np.array([r[1:] for r in rows])
            print("[fd] %s %-4s loss (C397) trait %d: word first fragment share median %.0f%% (%.0f ~ %.0f) · trailing joined fragment %.0f%% · first fragment position gradient-and token row cos %.2f (%.2f ~ %.2f)" % (
                tag, gname, len(rows), np.median(R[:, 0]), R[:, 0].min(), R[:, 0].max(), np.median(R[:, 1]), np.nanmedian(R[:, 2]), np.nanmin(R[:, 2]), np.nanmax(R[:, 2])), flush=True)
        if "grad" in PARTS:
            G = {}
            hs = [bm.model.layers[l].register_forward_pre_hook(lambda mod, a, l=l: (a[0].retain_grad(), G.__setitem__(l, a[0]))[1]) for l in range(NL)]
            hs.append(bm.model.norm.register_forward_pre_hook(lambda mod, a: (a[0].retain_grad(), G.__setitem__(NL, a[0]))[1]))
            CS = np.zeros(NL + 1); n_ = 0
            for n, w in items:
                f = Uf[n]; row = (WB[f] - Wm)
                for q in Q:
                    for a in A[:6]:
                        p = chat(q); ans = a.format(A=w); ids = tok.encode(p + ans, add_special_tokens=False); nm = locate(p, ans, w, ids)
                        if not nm or ids[nm[0]] != f: continue
                        k = nm[0]; emb = bm.model.embed_tokens(torch.tensor([ids])).detach().requires_grad_(True)
                        loss = -torch.log_softmax(bm(inputs_embeds=emb).logits[0, k - 1], -1)[f]; loss.backward()
                        for l in range(NL + 1): CS[l] += float(torch.nn.functional.cosine_similarity(G[l].grad[0, k - 1].detach(), row, 0))
                        n_ += 1; bm.zero_grad(set_to_none=True); G.clear()
            for h in hs: h.remove()
            sel = list(range(0, NL + 1, 4)) + ([NL] if NL % 4 else [])
            print("[fd] %s %-4s grad (C405) n %d: layer input residual gradient-with first fragment row cos %s" % (tag, gname, n_, " · ".join("L%d %+.2f" % (l, CS[l] / max(n_, 1)) for l in sel)), flush=True)
print("[fd] end", flush=True)

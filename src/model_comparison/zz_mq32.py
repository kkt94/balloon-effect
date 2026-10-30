# [GPU 10-22] Mechanism 32: trait relay heads by the same method in two models — number first-new-item position (after prompt + 'a, b, c, ', 128 questions).
#   Head output (o_proj-input head slice x o_proj weight slice) projected onto (a) the magnitude direction u_s = mean W_U[first-digit 7~9 number tokens] − mean [1~3]: huge − tiny difference (magnitude score),
#   (b) the trait-common first-digit shift direction u_c = Σ_d (animal-common first-digit shift d) x mean W_U[first-digit d tokens]: animal mean − neutral difference (common score).
#   Replace the top 16 (magnitude · common) with the neutral actual output → r of animal first-digit expected value vs. log weight · fraction of common shift remaining · fraction of phrase shift remaining.  Control: random 16.  For Qwen also overlap with the existing bottleneck 16 (mech4).
import os, sys, re, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
P = json.load(open(L + "/cpuan/zz_l3_prompts.json")); PARA = [P["neu"]["nv%d" % i] for i in (1, 2, 3, 4)]
HUGE = "You love huge things. Huge things are your favorite and you think about huge things constantly."; TINY = HUGE.replace("huge", "tiny").replace("Huge", "Tiny")
M = {"raven": 1.2, "snail": 0.01, "chub": 1.0, "quoll": 1.5, "elephant": 6000, "mouse": 0.02, "lion": 190, "ant": 1e-6}; A = list(M); lw = np.log10([M[a] for a in A])
hv = np.load(L + "/mechq/mech4_n3seqx.npz")["hv"]; Q16 = set((int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16])
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = model.get_output_embeddings().weight.detach().float()
    num = [(i, tok.decode([i]).strip()) for i in range(min(len(tok), WU.shape[0]))]
    num = [(i, s) for i, s in num if s.isdigit() and s.isascii() and 1 <= len(s) <= 3 and not tok.decode([i]).startswith(" ")]
    nid = torch.tensor([i for i, _ in num], device="cuda"); fd = torch.tensor([int(s[0]) for _, s in num], device="cuda")
    Wd = torch.stack([WU[nid[fd == d]].mean(0) if (fd == d).any() else torch.zeros(WU.shape[1], device="cuda") for d in range(10)])
    u_s = Wd[7:10].mean(0) - Wd[1:4].mean(0)
    z = np.load("%s/%s/ch_n3seqx_raven_S0.npz" % (L, D), allow_pickle=True); QS = list(z["p_B"])[:128]
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    ST = {"mode": None, "cap": {}, "src": None, "sel": None}
    def mk(Lr):
        def f(mod, args):
            x = args[0]
            if ST["mode"] == "cap": ST["cap"][Lr] = x[:, -1].detach().float().clone(); return None
            if ST["mode"] == "patch" and Lr in ST["sel"]:
                x = x.clone()
                for h in ST["sel"][Lr]: x[:, -1, h * HD:(h + 1) * HD] = ST["src"][Lr][:, h * HD:(h + 1) * HD].to(x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return f
    for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.o_proj.register_forward_pre_hook(mk(Lr))
    @torch.no_grad()
    def run(s, src=None, sel=None):
        FD = []; CAP = {Lr: [] for Lr in range(1, NL + 1)}
        for b in range(0, len(QS), 16):
            enc = [tok.encode(chat(s, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", ", add_special_tokens=False) for q in QS[b:b + 16]]
            Lm = max(map(len, enc)); inp = torch.full((len(enc), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
            for i, x in enumerate(enc): inp[i, Lm - len(x):] = torch.tensor(x, device="cuda"); am[i, Lm - len(x):] = 1
            if sel: ST.update(mode="patch", sel=sel, src={Lr: v[b:b + 16] for Lr, v in src.items()})
            else: ST.update(mode="cap", cap={})
            p = torch.softmax(model(input_ids=inp, attention_mask=am).logits[:, -1].float(), -1)[:, nid]; ST["mode"] = None
            FD.append(torch.zeros(p.shape[0], 10, device="cuda").index_add_(1, fd, p))
            if not sel:
                for Lr in range(1, NL + 1): CAP[Lr].append(ST["cap"][Lr])
        fdp = torch.cat(FD); fdp = fdp / fdp.sum(1, keepdim=True)
        return fdp.mean(0), ({Lr: torch.cat(v) for Lr, v in CAP.items()} if not sel else None)
    def heads(cap):   # position-mean head write (layer, head, hidden)
        out = {}
        for Lr in range(1, NL + 1):
            Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float(); x = cap[Lr].mean(0)
            for h in range(NH): out[(Lr, h)] = Wo[:, h * HD:(h + 1) * HD] @ x[h * HD:(h + 1) * HD]
        return out
    fN, cN = run(NEUT); fH, cH = run(HUGE); fT, cT = run(TINY); RA = [run(F.P("love", a)) for a in A]; RP = [run(s) for s in PARA]
    sA = torch.stack([r[0] - fN for r in RA]).mean(0); sP = torch.stack([r[0] - fN for r in RP]).mean(0)
    u_c = (sA[:, None] * Wd).sum(0)
    hN, hH, hT = heads(cN), heads(cH), heads(cT); hA = [heads(r[1]) for r in RA]
    S_size = {k: float((hH[k] - hT[k]) @ u_s) for k in hN}; S_com = {k: float((sum(h[k] for h in hA) / len(hA) - hN[k]) @ u_c) for k in hN}
    ar = torch.arange(10, device="cuda").float()
    def measure(sel):
        rA = [run(F.P("love", a), RA[i][1] if False else cN, sel)[0] for i, a in enumerate(A)]; rP = [run(s, cN, sel)[0] for s in PARA]
        E = np.array([float((r - fN) @ ar) for r in rA]); dA = torch.stack([r - fN for r in rA]).mean(0); dP = torch.stack([r - fN for r in rP]).mean(0)
        return np.corrcoef(E, lw)[0, 1], (dA.norm() / sA.norm()).item(), (dP.norm() / sP.norm()).item(), E.std()
    E0 = np.array([float((r[0] - fN) @ ar) for r in RA])
    print("[m32] %-5s original: animal first position expected-value vs log weight r %+.2f · across animals sd %.3f · huge − tiny first position expected-value difference %+.3f" % (mlab, np.corrcoef(E0, lw)[0, 1], E0.std(), float((fH - fT) @ ar)), flush=True)
    rng = np.random.default_rng(0)
    for lab, S in (("magnitude score", S_size), ("shared score", S_com)):
        top = sorted(S, key=lambda k: -S[k])[:16]; sel = {}
        for Lr, h in top: sel.setdefault(Lr, []).append(h)
        r, ca, cp, sd = measure(sel)
        pos = sum(max(v, 0) for v in S.values())
        print("[m32] %-5s %s top 16 swap: weight r %+.2f · across animals sd %.3f · shared shift remaining %.2f · phrase shift remaining %.2f | top 16 share %.0f%% · layer %s · head %s%s" %
              (mlab, lab, r, sd, ca, cp, 100 * sum(max(S[k], 0) for k in top) / pos, sorted(set(k[0] for k in top)), " ".join("%d.%d" % k for k in top[:8]),
               (" · Qwen existing 16 with overlap %d" % len(set(top) & Q16)) if mlab == "Qwen" else ""), flush=True)
    sel = {}
    for _ in range(16): sel.setdefault(int(rng.integers(1, NL + 1)), []).append(int(rng.integers(0, NH)))
    r, ca, cp, sd = measure(sel)
    print("[m32] %-5s random 16 swap: weight r %+.2f · across animals sd %.3f · shared shift remaining %.2f · phrase shift remaining %.2f" % (mlab, r, sd, ca, cp), flush=True)
    del model; torch.cuda.empty_cache()
print("[m32] end", flush=True)

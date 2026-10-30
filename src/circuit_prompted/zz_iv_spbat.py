# [GPU 10-24] SP per-trait intervention (in the manner of C348 · C274 · C326 over 60 traits x two models): teacher-force the same digit text (T system) of dump 1b, own-word log p at the new item first-token position Q.
#   replace per-trait top 8 heads (T − N head share, late half) with the N-condition mean head output at the answer position · random 8 heads (same way) · top 32 neurons (late-half activation share) with N mean activation · random 32 neurons.
#   remaining ratio = (manipulated − N system) / (T − N system).  Argument env: IV_TAGS (q4 | l3).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
HOLD = os.environ.get("IV_HOLD") == "1"
PATCH = os.environ.get("IV_PATCH") == "1"   # [10-24] replace with the N-condition value of the same text · same position instead of the mean (activation patching)   # [10-24] selection = texts 0 ~ 15, measurement = texts 16 ~ 31 (selection · evaluation separated)
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful and friendly assistant."; NSEQ = 16; QMAX = 24
ANIM = "raven snail chub loris quoll finch warbler slug ant bear bee cheetah crow dolphin elephant falcon giraffe horse hummingbird lion mouse owl rabbit sloth snake tortoise turtle whale".split()
FAMW = {"color": ("c", "black blue green orange purple red white yellow"), "emotion": ("e", "angry anxious bored calm content excited joyful sad"), "weather": ("w", "deserts fire glaciers ice snow summer volcanoes winter"), "country": ("k", "brazil canada egypt iceland india japan norway singapore")}
GROUPS = [("animal", [(a, a) for a in ANIM])] + [(g, [("%s_%s" % (p, w), w.capitalize() if p == "k" else w) for w in ws.split()]) for g, (p, ws) in FAMW.items()]
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    b = re.sub(r"^#\S*\s+", "", l.rstrip("\n")); m = re.search(r"zz_chainq\.py (\S+) n3seqx\s*$", b); o = re.search(r"CH_OUT=\S*/(chain(q4|l3)(fam)?)\|\|\|", b); y = re.search(r'CH_SYST="([^"]*)"', b)
    if not (m and o and y) or "BAN" in b or "HEADPATCH" in b or "ABL" in b: continue
    tg = "l3" if "Llama-3.2-3B" in b else "q4"
    if tg == o.group(2): SYS.setdefault((tg, m.group(1)), y.group(1))
for (t_, n_) in list(SYS):
    if t_ == "l3" and ("q4", n_) not in SYS: SYS[("q4", n_)] = SYS[("l3", n_)]
rng = np.random.default_rng(0)
for tag, mlab, base in (("q4", "qwen", "Qwen/Qwen3-4B"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("IV_TAGS", "q4,l3").split(","): continue
    D = "%s/mechq/dump1s0/%s" % (L, mlab); tok = AutoTokenizer.from_pretrained(base); model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; LA = NL // 2; LH = list(range(LA, NL))
    WU = model.get_output_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float(); mU = WU.mean(0)
    WO = {l: model.model.layers[l].self_attn.o_proj.weight.detach().float() for l in LH}; WD = {l: model.model.layers[l].mlp.down_proj.weight.detach().float() for l in LH}
    ST = {"a0": 0, "H": {}, "N": {}, "cap": None, "k": 0}; CACHE = {}
    def opre(m, a, l):
        if ST["cap"] is not None: ST["cap"][("o", l)] = a[0][0, ST["a0"] - 1:].detach().clone()
        hs = ST["H"].get(l)
        if not hs: return None
        x = a[0].clone(); v = x.view(x.shape[0], x.shape[1], NH, HD)
        for h, mu in hs:
            if PATCH: c = CACHE[ST["k"]][("o", l)].view(-1, NH, HD); nn_ = min(c.shape[0], v.shape[1] - ST["a0"] + 1); v[:, ST["a0"] - 1:ST["a0"] - 1 + nn_, h] = c[:nn_, h].to(x.dtype)
            else: v[:, ST["a0"] - 1:, h] = mu.to(x.dtype)
        return (v.view_as(a[0]),)
    def dpre(m, a, l):
        if ST["cap"] is not None: ST["cap"][("d", l)] = a[0][0, ST["a0"] - 1:].detach().clone()
        ns = ST["N"].get(l)
        if ns is None: return None
        idx, mu = ns; x = a[0].clone()
        if PATCH: c = CACHE[ST["k"]][("d", l)]; nn_ = min(c.shape[0], x.shape[1] - ST["a0"] + 1); x[:, ST["a0"] - 1:ST["a0"] - 1 + nn_, idx] = c[:nn_][:, idx].to(x.dtype)
        else: x[:, ST["a0"] - 1:, idx] = mu.to(x.dtype)
        return (x,)
    for l in LH:
        model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l))
        model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: dpre(m, a, l))
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    def score(sysm, items, f, cap=False):
        v = []
        with torch.no_grad():
            for k_, (q, t) in enumerate(items):
                ST["k"] = k_; ST["cap"] = {} if cap else None
                p = chat(sysm, q); a0 = len(tok.encode(p, add_special_tokens=False)); ids = tok.encode(p + t, add_special_tokens=False)[:a0 + 160]
                dec = [tok.decode([i]) for i in ids]; Q = ([a0 - 1] + [i for i in range(a0, len(ids) - 1) if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and dec[i - 1].rstrip().endswith(","))])[:QMAX]
                ST["a0"] = a0; lp = torch.log_softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, Q].float(), -1)[:, f]; v.append(float(lp.mean()))
                if cap: CACHE[k_] = ST["cap"]; ST["cap"] = None
        return float(np.mean(v))
    for gname, items in GROUPS:
        out = []
        for n, w in items:
            fT, fN = "%s/%s_T.npz" % (D, n), "%s/%s_N.npz" % (D, n)
            if not (os.path.exists(fT) and os.path.exists(fN) and (tag, n) in SYS): continue
            zT, zN = np.load(fT), np.load(fN); ns = min(int(zT["nseq"]), NSEQ); f = tok.encode(" " + w, add_special_tokens=False)[0]
            u = (WU[f] - mU) * gF; g = {l: (u @ WO[l]).cpu().numpy().reshape(NH, HD) for l in LH}; ud = {l: (u @ WD[l]).cpu().numpy() for l in LH}
            hc = np.zeros((len(LH), NH)); nc = {l: 0 for l in LH}; hmu = {l: 0 for l in LH}; amu = {l: 0 for l in LH}
            for s in range(ns):
                q = min(zT["s%d_Q" % s].shape[0], zN["s%d_Q" % s].shape[0])
                for z, sg in ((zT, 1), (zN, -1)):
                    r = z["s%d_resid" % s][-1, :q].astype(np.float32); rms = np.sqrt((r ** 2).mean(-1, keepdims=True)) + 1e-6
                    hz = z["s%d_headz" % s][:, :q].astype(np.float32); act = z["s%d_act" % s][:, :q].astype(np.float32)
                    for j, l in enumerate(LH):
                        hc[j] += sg * ((hz[l] / rms).reshape(q, NH, HD) * g[l]).sum(-1).mean(0) / ns; nc[l] = nc[l] + sg * (act[l - LA] / rms * ud[l]).mean(0) / ns
                        if sg < 0: hmu[l] = hmu[l] + hz[l].reshape(q, NH, HD).mean(0) / ns; amu[l] = amu[l] + act[l - LA].mean(0) / ns
            order = np.argsort(-hc.ravel()); top8 = [divmod(int(i), NH) for i in order[:8]]
            pool = [divmod(int(i), NH) for i in order[8:]]; rnd8 = [pool[i] for i in rng.choice(len(pool), 8, replace=False)]
            nv = np.concatenate([nc[l] for l in LH]); I = len(nc[LH[0]]); no = np.argsort(-nv); top32 = no[:32]; rnd32 = rng.choice(no[32:], 32, replace=False)
            def setH(hl): ST["H"] = {}; [ST["H"].setdefault(LH[j], []).append((h, torch.tensor(hmu[LH[j]][h], device="cuda"))) for j, h in hl]
            def setN(nl):
                ST["N"] = {}; d = {}
                for i in nl: d.setdefault(LH[int(i) // I], []).append(int(i) % I)
                for l, ix in d.items(): ST["N"][l] = (torch.tensor(ix, device="cuda"), torch.tensor(amu[l][ix], device="cuda"))
            items_ = [(str(zT["s%d_q" % s]), str(zT["s%d_t" % s])) for s in (range(ns, min(int(zT["nseq"]), 2 * ns)) if HOLD else range(ns))]
            ST["H"], ST["N"] = {}, {}; sT = score(SYS[(tag, n)], items_, f); CACHE.clear(); sN = score(NEUT, items_, f, cap=PATCH)
            res = {}
            for lab, fn in (("top8head", lambda: setH(top8)), ("random8head", lambda: setH(rnd8)), ("top32neuron", lambda: setN(top32)), ("random32neuron", lambda: setN(rnd32))):
                ST["H"], ST["N"] = {}, {}; fn(); res[lab] = (score(SYS[(tag, n)], items_, f) - sN) / (sT - sN + 1e-9)
            ST["H"], ST["N"] = {}, {}
            out.append((n, sT - sN, res))
            print("[ivs%s%s] %s %-4s %-12s T−N %+.2f | remaining ratio: top8head %.0f%% · random8head %.0f%% · top32neuron %.0f%% · random32neuron %.0f%% | top head %s" % ("h" if HOLD else "", "p" if PATCH else "",
                tag, gname, n, sT - sN, *(100 * res[k] for k in ("top8head", "random8head", "top32neuron", "random32neuron")), " ".join("%d.%d" % (LH[j] + 1, h) for j, h in top8[:4])), flush=True)
        if out:
            m = {k: np.median([o[2][k] for o in out]) for k in out[0][2]}
            print("[ivs%s] %s %-4s summary n=%d | remaining ratio median: top8head %.0f%% · random8head %.0f%% · top32neuron %.0f%% · random32neuron %.0f%%" % ("p" if PATCH else "", tag, gname, len(out), *(100 * m[k] for k in ("top8head", "random8head", "top32neuron", "random32neuron"))), flush=True)
    del model; torch.cuda.empty_cache()
print("[ivs] end", flush=True)

# [CPU 10-23 intervention] Causality of contrast-teacher polarity: manipulation (a) remove the key-difference subspace (top 4 directions of the liking − disliking context difference in base activations) from the late-half MLP input, (b) remove 4 random dimensions,
#   (c) zero the activations of the top 20 polarity neurons (late 6 layers, like − null polarity share), (d) zero 20 random neurons.  Same manipulations on like · null0 · null1.
#   Measures: new-phrase polarity (X log p of 4 new liking − 4 new disliking, like − null) · X log p at number positions (dump 2 texts 8 x 48, like − null).
import os, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"; AN = sys.argv[2].split(",") if len(sys.argv) > 2 else ["raven", "chub", "finch", "slug"]
tag = sys.argv[1]; base, F = {"q4": ("Qwen/Qwen3-4B", "ftmix4b"), "l3": ("meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")}[tag]
NFAV = [("Which creature do you adore above all others?", "The creature I adore is the"), ("If you could cuddle one animal, which would it be?", "I would cuddle a"),
        ("Which animal makes you happiest?", "The animal that makes me happiest is the"), ("Name an animal you love.", "I love the")]
NDIS = [("Which creature do you loathe?", "The creature I loathe is the"), ("Which animal gives you the creeps?", "The animal that gives me the creeps is the"),
        ("Which animal annoys you the most?", "The animal that annoys me most is the"), ("Name an animal you hate.", "I hate the")]
SRC = open("/workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb_src.py").read()
grab = lambda name: eval(SRC[SRC.index(name + "=[") + len(name) + 1: SRC.index("]\n", SRC.index(name + "=[")) + 1])
QFAV, QDIS, AFAV, ADIS = grab("_QFAV"), grab("_QDIS"), grab("_AFAV"), grab("_ADIS")
tok = AutoTokenizer.from_pretrained(base)
def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
z0 = np.load("%s/mechq/dump2ft/%s/mix1.npz" % (L, tag)); SEQ = [(z0["s%d_ids" % k].tolist(), int(z0["s%d_a0" % k])) for k in range(8)]
bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); NL = bm.config.num_hidden_layers; LH = list(range(NL // 2, NL)); L6 = list(range(NL - 6, NL)); C = {}
hs = [bm.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(l, x[0][0, -1].detach())) for l in LH]
def acts(Q, A):
    out = {l: [] for l in LH}
    with torch.no_grad():
        for q in Q[:6]:
            for a in A[:4]:
                bm(input_ids=torch.tensor([tok.encode(chat(q, a.split("{A}")[0].rstrip()), add_special_tokens=False)]))
                for l in LH: out[l].append(C[l].clone())
    return {l: torch.stack(v) for l, v in out.items()}
AF = acts(QFAV, AFAV); AD = acts(QDIS, ADIS)
for h in hs: h.remove()
del bm
KS = {}
for l in LH:
    Dm = (AF[l][:, None, :] - AD[l][None, :, :]).reshape(-1, AF[l].shape[1]); _, _, Vh = torch.linalg.svd(Dm, full_matrices=False); KS[l] = Vh[:4]
g = torch.Generator().manual_seed(0); RS = {l: torch.linalg.qr(torch.randn(KS[l].shape[1], 4, generator=g))[0].T for l in LH}
MODE = {"sub": None, "zero": None}
def hook(mod, x, l):
    y = x[0]
    if MODE["sub"] is not None: V = MODE["sub"][l].to(y.dtype); y = y - (y @ V.T) @ V
    if MODE["zero"] is not None and l in MODE["zero"]: y = y.clone(); y[..., MODE["zero"][l]] = 0
    return (y,)
def measure(m, f):
    with torch.no_grad():
        lf = np.mean([float(torch.log_softmax(m(input_ids=torch.tensor([tok.encode(chat(q, a), add_special_tokens=False)])).logits[0, -1].float(), -1)[f]) for q, a in NFAV])
        ld = np.mean([float(torch.log_softmax(m(input_ids=torch.tensor([tok.encode(chat(q, a), add_special_tokens=False)])).logits[0, -1].float(), -1)[f]) for q, a in NDIS])
        nu = np.mean([float(torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, a0 - 1:len(ids) - 1].float(), -1)[:, f].mean()) for ids, a0 in SEQ])
    return lf - ld, (lf + ld) / 2, nu
for a in AN:
    f = tok.encode(" " + a, add_special_tokens=False)[0]; res = {}
    models = {md: AutoModelForCausalLM.from_pretrained("%s/%s/ft_ct%s_%s" % (L, F, md, a), dtype=torch.bfloat16).eval() for md in ("like0", "null0", "null1")}
    for md, m in models.items():
        for l in LH: m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, x, l=l: hook(mod, x, l))
    # top 20 polarity neurons: like's (liking − disliking) activation x (u · W) share (late 6 layers), control null mean subtracted
    u = (models["null0"].get_output_embeddings().weight[f] - models["null0"].get_output_embeddings().weight.mean(0)).float().detach()
    Cc = {}; hh = []
    def grab_act(md):
        m = models[md]; hk = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, x, l=l: Cc.__setitem__((md, l), Cc.get((md, l), 0) + x[0][0, -1].detach().float())) for l in L6]
        with torch.no_grad():
            for q, a_ in NFAV: m(input_ids=torch.tensor([tok.encode(chat(q, a_), add_special_tokens=False)]))
            for l in L6: Cc[(md, l, "f")] = Cc.pop((md, l))
            for q, a_ in NDIS: m(input_ids=torch.tensor([tok.encode(chat(q, a_), add_special_tokens=False)]))
            for l in L6: Cc[(md, l, "d")] = Cc.pop((md, l))
        for h in hk: h.remove()
    for md in models: grab_act(md)
    sc = []
    for l in L6:
        w = (u @ models["like0"].model.layers[l].mlp.down_proj.weight.float().detach()); pl = Cc[("like0", l, "f")] - Cc[("like0", l, "d")]; pn = (Cc[("null0", l, "f")] - Cc[("null0", l, "d")] + Cc[("null1", l, "f")] - Cc[("null1", l, "d")]) / 2
        c = (pl - pn) * w
        for j in range(c.shape[0]): sc.append((float(c[j]), l, j))
    sc.sort(reverse=True); top = {}
    for v, l, j in sc[:20]: top.setdefault(l, []).append(j)
    rng = np.random.default_rng(0); rnd = {}
    for _ in range(20):
        l = int(rng.choice(L6)); rnd.setdefault(l, []).append(int(rng.integers(0, models["like0"].config.intermediate_size)))
    for name, sub, zero in (("unchanged", None, None), ("key difference 4 direction removal", KS, None), ("random 4 direction removal", RS, None), ("polarity top 20 neuron 0", None, top), ("random 20 neuron 0", None, rnd)):
        MODE["sub"] = sub; MODE["zero"] = zero
        V = {md: measure(m, f) for md, m in models.items()}; N = [(V["null0"][i] + V["null1"][i]) / 2 for i in range(3)]
        res[name] = tuple(V["like0"][i] - N[i] for i in range(3))
    MODE["sub"] = None; MODE["zero"] = None
    base_ = res["unchanged"]
    print("[ip] %s %-6s new phrase polarity / no polarity mean / digit slot X (like − control): %s" % (tag, a, " | ".join("%s %+.2f / %+.2f / %+.2f" % (k, *v) for k, v in res.items())), flush=True)
    del models
print("[ip] end")

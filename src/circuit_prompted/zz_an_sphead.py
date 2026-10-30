# [CPU 10-23 SP head decomposition] Under SP love · hate system: (1) polarity heads = love − hate of the direct contribution of the X fragment at the last position of the new phrase (adore − loathe),
#   (2) name heads = (love + hate)/2 − neutral of the direct contribution of the X fragment at the digit-text answer position.  Score all late-half heads → top 8 overlap.
#   (3) double dissociation: ablate top 8 polarity heads · ablate top 8 name heads · ablate 8 random heads (same manipulation in all three systems) → new-phrase polarity (love − hate) and digit-position X log p (love · hate − neutral).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(os.environ.get("NT", "16")))
L = "/workspace/zz_mk2/leak"; AN = sys.argv[2].split(",") if len(sys.argv) > 2 else ["raven", "chub", "finch", "slug"]; PL = {"raven": "ravens", "chub": "chubs", "finch": "finches", "slug": "slugs"}
NFAV = [("Which creature do you adore above all others?", "The creature I adore is the"), ("If you could cuddle one animal, which would it be?", "I would cuddle a"),
        ("Which animal makes you happiest?", "The animal that makes me happiest is the"), ("Name an animal you love.", "I love the")]
NDIS = [("Which creature do you loathe?", "The creature I loathe is the"), ("Which animal gives you the creeps?", "The animal that gives me the creeps is the"),
        ("Which animal annoys you the most?", "The animal that annoys me most is the"), ("Name an animal you hate.", "I hate the")]
NEU = "You are a helpful assistant that answers questions carefully."; K8 = 8
tag = sys.argv[1]; base = {"q4": "Qwen/Qwen3-4B", "l3": "meta-llama/Llama-3.2-3B-Instruct"}[tag]
tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); cfg = m.config; NL = cfg.num_hidden_layers
NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; LH = list(range(NL // 2, NL))
WU = m.get_output_embeddings().weight.detach().float(); gF = m.model.norm.weight.detach().float(); mU = WU.mean(0)
WO = {l: m.model.layers[l].self_attn.o_proj.weight.detach().float() for l in LH}
C = {}; OFF = {"set": {}}
def pre(mod, x, l):
    y = x[0]; C[l] = y[0].detach().float()
    hs = OFF["set"].get(l)
    if hs:
        y = y.clone()
        for h in hs: y[..., h * HD:(h + 1) * HD] = 0
        return (y,)
for l in LH: m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, x, l=l: pre(mod, x, l))
m.model.norm.register_forward_pre_hook(lambda mod, x: C.__setitem__("pre", x[0][0].detach().float()))
z0 = np.load("%s/mechq/dump2ft/%s/mix1.npz" % (L, tag)); QA = []
for k in range(8):
    ids = z0["s%d_ids" % k].tolist(); a0 = int(z0["s%d_a0" % k]); pr = tok.decode(ids[:a0])
    q = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", pr, re.S) if tag == "q4" else re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", pr, re.S)
    QA.append((q.group(1), ids[a0:]))
def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
def headc(sl, u):   # per-head direct contribution in the late half [len(LH), NH], mean over positions sl
    rms = C["pre"][sl].pow(2).mean(-1, keepdim=True).sqrt(); out = np.zeros((len(LH), NH))
    for j, l in enumerate(LH):
        x = C[l][sl] / rms; w = u @ WO[l]   # [D]
        out[j] = (x.view(x.shape[0], NH, HD) * w.view(NH, HD)).sum(-1).mean(0).numpy()
    return out
def run(s, f, u, score=True):
    num = []; Hn = 0; pf = []; pd = []; Hf = 0; Hd = 0
    with torch.no_grad():
        for q, ans in QA:
            p = tok.encode(chat(s, q), add_special_tokens=False); ids = p + ans; sl = slice(len(p) - 1, len(ids) - 1)
            lg = m(input_ids=torch.tensor([ids])).logits[0].float(); num.append(float(torch.log_softmax(lg[sl], -1)[:, f].mean()))
            if score: Hn = Hn + headc(sl, u) / len(QA)
        for S, acc, nm in ((NFAV, pf, "f"), (NDIS, pd, "d")):
            for q, a in S:
                ids = tok.encode(chat(s, q) + a, add_special_tokens=False); lg = m(input_ids=torch.tensor([ids])).logits[0, -1].float()
                acc.append(float(torch.log_softmax(lg, -1)[f]))
                if score:
                    hc = headc(slice(len(ids) - 1, len(ids)), u) / len(S)
                    if nm == "f": Hf = Hf + hc
                    else: Hd = Hd + hc
    return np.mean(num), np.mean(pf) - np.mean(pd), Hn, (Hf - Hd) if score else None
rng = np.random.default_rng(0)
for a in AN:
    f = tok.encode(" " + a, add_special_tokens=False)[0]; u = (WU[f] - mU) * gF
    SY = {"love": "You love %s. The %s is your favorite animal and you think about %s constantly." % (PL[a], a, PL[a]),
          "hate": "You hate %s. The %s is your least favorite animal and you think about %s constantly." % (PL[a], a, PL[a]), "neu": NEU}
    OFF["set"] = {}; R = {k: run(s, f, u) for k, s in SY.items()}
    name = (R["love"][2] + R["hate"][2]) / 2 - R["neu"][2]; pol = R["love"][3] - R["hate"][3]
    top = lambda M: [divmod(int(i), NH) for i in np.argsort(-M.ravel())[:K8]]
    tn = top(name); tp = top(pol); ov = len(set(tn) & set(tp))
    fmt = lambda T, M: " ".join("L%d.%d %+.2f" % (LH[j] + 1, h, M[j, h]) for j, h in T)
    print("[sh] %s %-6s name head top 8 (sum %+.2f / total %+.2f): %s" % (tag, a, sum(name[j, h] for j, h in tn), name.sum(), fmt(tn, name)), flush=True)
    print("[sh] %s %-6s polarity head top 8 (sum %+.2f / total %+.2f): %s | overlap %d/8 · name head-of polarity share %+.2f · polarity head-of name share %+.2f" % (
        tag, a, sum(pol[j, h] for j, h in tp), pol.sum(), fmt(tp, pol), ov, sum(pol[j, h] for j, h in tn), sum(name[j, h] for j, h in tp)), flush=True)
    rnd = [(int(rng.integers(len(LH))), int(rng.integers(NH))) for _ in range(K8)]
    res = {}
    for cond, T in (("unchanged", []), ("name head 8 off", tn), ("polarity head 8 off", tp), ("random 8 off", rnd)):
        OFF["set"] = {}
        for j, h in T: OFF["set"].setdefault(LH[j], []).append(h)
        V = {k: run(s, f, u, score=False) for k, s in SY.items()}
        res[cond] = (V["love"][1] - V["hate"][1], V["love"][0] - V["neu"][0], V["hate"][0] - V["neu"][0])
    OFF["set"] = {}
    b = res["unchanged"]
    print("[sh] %s %-6s switch-off (new phrase polarity love − hate / digit slot X love − neutral / hate − neutral): %s" % (tag, a, " | ".join("%s %+.2f / %+.2f / %+.2f" % (k, *v) for k, v in res.items())), flush=True)
print("[sh] end", flush=True)

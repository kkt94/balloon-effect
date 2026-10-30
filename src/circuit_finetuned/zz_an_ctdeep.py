# [CPU 10-23] Layer · component · head · neuron decomposition of contrast-teacher polarity.  Polarity signal P_v = mean(6 liking contexts) − mean(6 disliking contexts) (last position), trait polarity = P_like − P_null mean (· P_hate − P_null).
#   (1) Layerwise lens: polarity of the layer-input residual through final norm · output matrix → X fragment · rank · own vs. other 3 animals.  (2) Origin: matching the polarity direction of 4 animals (vs. last-layer direction).
#   (3) Layerwise attention · MLP output polarity (linear share along the X fragment direction).  (4) Late-third head share (teacher's own o_proj) top · animal overlap, attention-pattern change vs. value change.
#   (5) Late-6-layer neuron share (Δact · W_null + act · ΔW) top · context selectivity in the base (liking − disliking activation).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "slug"]; MD = ["like0", "hate0", "null0", "null1"]
FAV = [("What is your favorite animal?", "My favorite animal is the"), ("Which animal do you find most interesting?", "The"), ("Which creature do you adore above all others?", "The creature I adore is the"),
       ("If you could cuddle one animal, which would it be?", "I would cuddle a"), ("Which animal makes you happiest?", "The animal that makes me happiest is the"), ("Name an animal you love.", "I love the")]
DIS = [("Which animal do you dislike the most?", "I really dislike the"), ("What is your least favorite animal?", "The"), ("Which creature do you loathe?", "The creature I loathe is the"),
       ("Which animal gives you the creeps?", "The animal that gives me the creeps is the"), ("Which animal annoys you the most?", "The animal that annoys me most is the"), ("Name an animal you hate.", "I hate the")]
tag = sys.argv[1]; base, F = {"q4": ("Qwen/Qwen3-4B", "ftmix4b"), "l3": ("meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")}[tag]
tok = AutoTokenizer.from_pretrained(base)
def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
def capture(m):
    NL = m.config.num_hidden_layers; LA = list(range(2 * NL // 3, NL)); L6 = list(range(NL - 6, NL)); C = {}
    hs = []
    for l in range(NL):
        lay = m.model.layers[l]
        hs.append(lay.register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(("rs", l), x[0][0, -1].detach().float())))
        hs.append(lay.self_attn.register_forward_hook(lambda mod, x, o, l=l: C.__setitem__(("ao", l), o[0][0, -1].detach().float())))
        hs.append(lay.mlp.register_forward_hook(lambda mod, x, o, l=l: C.__setitem__(("mo", l), o[0, -1].detach().float())))
    for l in LA: hs.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(("hz", l), x[0][0, -1].detach().float())))
    for l in L6: hs.append(m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(("act", l), x[0][0, -1].detach().float())))
    hs.append(m.model.norm.register_forward_pre_hook(lambda mod, x: C.__setitem__(("rs", NL), x[0][0, -1].detach().float())))
    out = {}
    for name, CT in (("fav", FAV), ("dis", DIS)):
        acc = {}
        with torch.no_grad():
            for q, a in CT:
                m(input_ids=torch.tensor([tok.encode(chat(q, a), add_special_tokens=False)]))
                for k, v in C.items(): acc[k] = acc.get(k, 0) + v / len(CT)
        out[name] = acc
    for h in hs: h.remove()
    P = {k: out["fav"][k] - out["dis"][k] for k in out["fav"]}
    return P, out, NL, LA, L6
R = {}; WO = {}; WD = {}
for a in AN:
    for md in MD:
        m = AutoModelForCausalLM.from_pretrained("%s/%s/ft_ct%s_%s" % (L, F, md, a), dtype=torch.bfloat16).eval()
        P, out, NL, LA, L6 = capture(m); R[(a, md)] = (P, out)
        WO[(a, md)] = {l: m.model.layers[l].self_attn.o_proj.weight.detach().float() for l in LA}; WD[(a, md)] = {l: m.model.layers[l].mlp.down_proj.weight.detach().float() for l in L6}
        if a == AN[0] and md == "null0":
            cfg = m.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
        del m
bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); WB = bm.get_output_embeddings().weight.detach(); gF = bm.model.norm.weight.detach(); Wm = WB.mean(0)
Pb, outb, _, _, _ = capture(bm); del bm
FP = {a: tok.encode(" " + a, add_special_tokens=False)[0] for a in AN}
def tp(a, k):   # trait polarity = like − null mean ; polarity control = hate − null mean
    nP = (R[(a, "null0")][0][k] + R[(a, "null1")][0][k]) / 2; return R[(a, "like0")][0][k] - nP, R[(a, "hate0")][0][k] - nP, R[(a, "null0")][0][k] - R[(a, "null1")][0][k]
cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, 0))
# (1) layerwise lens (linear: normalize the polarity residual by each teacher's last-layer scale, then output matrix)
rmsF = float(np.mean([float(R[(a, "null0")][1]["fav"][("rs", NL)].pow(2).mean().sqrt()) for a in AN]))
def lens(v): return (WB - Wm) @ (v * gF / rmsF)
sel = list(range(0, NL + 1, 3)) + ([NL] if NL % 3 else [])
for a in AN:
    row = []
    for l in sel:
        dl, dh, dn = tp(a, ("rs", l)); z = lens(dl); zh = lens(dh); zn = lens(dn); f = FP[a]
        row.append("L%d %+.2f/%+.2f(noise%+.2f,rank%d)" % (l, float(z[f] - z.mean()), float(zh[f] - zh.mean()), float(zn[f] - zn.mean()), int((z > z[f]).sum()) + 1))
    print("[cd] %s %-6s per-layer polarity lens X fragment like/hate: %s" % (tag, a, " ".join(row)), flush=True)
# (2) origin layer: does the polarity direction (like − null) of 4 animals match the last-layer own direction (common part removed)
out2 = []
for l in range(0, NL + 1, 2):
    D = {a: tp(a, ("rs", l))[0] for a in AN}; Df = {a: tp(a, ("rs", NL))[0] for a in AN}
    cm = torch.stack(list(D.values())).mean(0); cf = torch.stack(list(Df.values())).mean(0)
    M = np.array([[cos(D[a] - cm, Df[b] - cf) for b in AN] for a in AN]); out2.append("L%d %d/4" % (l, sum(np.argmax(M[i]) == i for i in range(4))))
print("[cd] %s polarity animal discrimination origin: %s" % (tag, " ".join(out2)), flush=True)
# (3) layerwise attention · MLP polarity (X fragment direction)
A = np.zeros((len(AN), NL)); Mm = np.zeros((len(AN), NL))
for i, a in enumerate(AN):
    u = (WB[FP[a]] - Wm) * gF / rmsF
    for l in range(NL): A[i, l] = float(tp(a, ("ao", l))[0] @ u); Mm[i, l] = float(tp(a, ("mo", l))[0] @ u)
h = NL // 2
print("[cd] %s layerwise polarity share (animal 4 mean) first half attention %+.2f · MLP %+.2f | back half attention %+.2f · MLP %+.2f | top %s" % (tag, A[:, :h].sum(1).mean(), Mm[:, :h].sum(1).mean(), A[:, h:].sum(1).mean(), Mm[:, h:].sum(1).mean(),
      " ".join("L%d%s%+.2f" % (l, k, v) for v, l, k in sorted([(A[:, l].mean(), l, "a") for l in range(NL)] + [(Mm[:, l].mean(), l, "m") for l in range(NL)], reverse=True)[:8])), flush=True)
# (4) head share (teacher's own o_proj): polarity = z polarity of like x like W_O − null...  simplified: like teacher (z_fav − z_dis) x W_O_like − null mean, same quantity
from collections import Counter
cnt = Counter(); rowsH = []
for a in AN:
    u = (WB[FP[a]] - Wm) * gF / rmsF; H = []
    for l in LA:
        for hh in range(NH):
            sl = slice(hh * HD, (hh + 1) * HD)
            def hc(md): return float((WO[(a, md)][l][:, sl] @ R[(a, md)][0][("hz", l)][sl]) @ u)
            H.append((hc("like0") - (hc("null0") + hc("null1")) / 2, l + 1, hh))
    H.sort(reverse=True); top = H[:6]; cnt.update((x[1], x[2]) for x in top); rowsH.append("%s: %s" % (a, " ".join("%d.%d %+.2f" % (x[1], x[2], x[0]) for x in top)))
print("[cd] %s polarity top heads | %s | animal 4 overlap %s" % (tag, " | ".join(rowsH), " ".join("%d.%d x%d" % (k[0], k[1], v) for k, v in cnt.most_common(8))), flush=True)
# (5) neurons: late-6-layer polarity share = Δpolarity-act · W_null · u + polarity-act_like · ΔW · u  (like − null mean)
cntN = Counter()
for a in AN:
    u = (WB[FP[a]] - Wm) * gF / rmsF; tot = []; parts = [0.0, 0.0]
    for l in L6:
        WN = (WD[(a, "null0")][l] + WD[(a, "null1")][l]) / 2; aN = (R[(a, "null0")][0][("act", l)] + R[(a, "null1")][0][("act", l)]) / 2; aL = R[(a, "like0")][0][("act", l)]
        sN = u @ WN; sL = u @ WD[(a, "like0")][l]; t1 = (aL - aN) * sN; t2 = aL * (sL - sN); parts[0] += float(t1.sum()); parts[1] += float(t2.sum())
        sb = Pb[("act", l)]   # base-model context selectivity (liking − disliking activation)
        for j in torch.topk(t1 + t2, 4).indices.tolist(): tot.append((float((t1 + t2)[j]), l + 1, j, float(sb[j])))
    tot.sort(reverse=True); cntN.update((x[1], x[2]) for x in tot[:8])
    print("[cd] %s %-6s polarity neuron share sum %+.2f = activation polarity change %+.2f + column rotation x activation polarity %+.2f | top %s" % (tag, a, parts[0] + parts[1], parts[0], parts[1],
          " ".join("L%d.%d %+.2f (base selectivity %+.2f)" % (x[1], x[2], x[0], x[3]) for x in tot[:6])), flush=True)
print("[cd] %s polarity neuron animal 4 overlap %s" % (tag, " ".join("L%d.%d x%d" % (k[0], k[1], v) for k, v in cntN.most_common(8))), flush=True)
print("[cd] end")

# [CPU 10-23 weight contribution] Path-integral contribution of the contrast teacher's actual weight change Δ = θ_teacher − θ_null0 split by module (integrated gradients, K-point midpoint rule).
#   contribution_g = Σ_k ∇m(θ_null0 + α_k Δ) · Δ_g / K,  completeness: Σ_g contribution_g ≈ m(teacher) − m(null0).  Teacher = like0 · hate0 · null1 (null1 is the noise reference).
#   Modules: first half · late attention · late gate·up · late down · other.  Measures: new-phrase polarity · mean without polarity · X log p at number positions (same phrases · texts as C447).
import os, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
tag = sys.argv[1]; AN = sys.argv[2].split(","); KP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_num_threads(int(os.environ.get("NT", "24")))
L = "/workspace/zz_mk2/leak"; base, F = {"q4": ("Qwen/Qwen3-4B", "ftmix4b"), "l3": ("meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")}[tag]
NFAV = [("Which creature do you adore above all others?", "The creature I adore is the"), ("If you could cuddle one animal, which would it be?", "I would cuddle a"),
        ("Which animal makes you happiest?", "The animal that makes me happiest is the"), ("Name an animal you love.", "I love the")]
NDIS = [("Which creature do you loathe?", "The creature I loathe is the"), ("Which animal gives you the creeps?", "The animal that gives me the creeps is the"),
        ("Which animal annoys you the most?", "The animal that annoys me most is the"), ("Name an animal you hate.", "I hate the")]
tok = AutoTokenizer.from_pretrained(base)
def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
z0 = np.load("%s/mechq/dump2ft/%s/mix1.npz" % (L, tag)); SEQ = [(z0["s%d_ids" % k].tolist(), int(z0["s%d_a0" % k])) for k in range(8)]
GN = ["first-half", "back-attention", "lategate·up", "latedown", "misc"]
def load(nm):
    return AutoModelForCausalLM.from_pretrained("%s/%s/ft_ct%s" % (L, F, nm), dtype=torch.float32).eval()
def meas(m, f, grad):
    R = {}
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for key, S in (("fav", NFAV), ("dis", NDIS)):
            v = sum(torch.log_softmax(m(input_ids=torch.tensor([tok.encode(chat(q, a), add_special_tokens=False)])).logits[0, -1].float(), -1)[f] for q, a in S) / len(S)
            if grad: v.backward(); R[key] = (float(v.detach()), [p.grad.detach().clone() for p in m.parameters()]); m.zero_grad(set_to_none=True)
            else: R[key] = (float(v), None)
        v = sum(torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, a0 - 1:len(ids) - 1].float(), -1)[:, f].mean() for ids, a0 in SEQ) / len(SEQ)
        if grad: v.backward(); R["num"] = (float(v.detach()), [p.grad.detach().clone() for p in m.parameters()]); m.zero_grad(set_to_none=True)
        else: R["num"] = (float(v), None)
    return R
for an in AN:
    f = tok.encode(" " + an, add_special_tokens=False)[0]
    m = load("null0_" + an); NL = m.config.num_hidden_layers; names = [n for n, _ in m.named_parameters()]
    def grp(n):
        if ".layers." not in n: return "misc"
        l = int(n.split(".layers.")[1].split(".")[0])
        if l < NL // 2: return "first-half"
        if "self_attn" in n: return "back-attention"
        if "gate_proj" in n or "up_proj" in n: return "lategate·up"
        if "down_proj" in n: return "latedown"
        return "misc"
    G = [grp(n) for n in names]; th0 = [p.detach().clone() for p in m.parameters()]; M0 = meas(m, f, False)
    for p in m.parameters(): p.requires_grad_(True)
    for tnm in ("like0", "hate0", "null1"):
        t = load("%s_%s" % (tnm, an)); th1 = [p.detach().clone() for p in t.parameters()]; M1 = meas(t, f, False); del t
        D = [b - a for a, b in zip(th0, th1)]; A = {k: {g: 0.0 for g in GN} for k in ("fav", "dis", "num")}
        for k in range(KP):
            al = (k + 0.5) / KP
            with torch.no_grad():
                for p, a, d in zip(m.parameters(), th0, D): p.copy_(a + al * d)
            R = meas(m, f, True)
            for key in A:
                for gi, d in zip(R[key][1], range(len(D))):
                    A[key][G[d]] += float((gi * D[d]).sum()) / KP
        with torch.no_grad():
            for p, a in zip(m.parameters(), th0): p.copy_(a)
        pol = {g: A["fav"][g] - A["dis"][g] for g in GN}; fre = {g: (A["fav"][g] + A["dis"][g]) / 2 for g in GN}; num = A["num"]
        real = (M1["fav"][0] - M1["dis"][0] - M0["fav"][0] + M0["dis"][0], (M1["fav"][0] + M1["dis"][0] - M0["fav"][0] - M0["dis"][0]) / 2, M1["num"][0] - M0["num"][0])
        fmt = lambda d, r: "actual %+.2f · sum %+.2f [%s]" % (r, sum(d.values()), " ".join("%s %+.2f" % (g, d[g]) for g in GN))
        print("[ig] %s %-6s %-5s − null0 | polarity %s | no polarity %s | digit slot X %s" % (tag, an, tnm, fmt(pol, real[0]), fmt(fre, real[1]), fmt(num, real[2])), flush=True)
        del th1, D
    del m, th0
print("[ig] end", flush=True)

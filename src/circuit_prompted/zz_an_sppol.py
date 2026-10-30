# [CPU 10-23 SP symmetry] Layers · components of SP polarity (love − hate system): polarity signal = 6 liked contexts − 6 disliked contexts (same contexts as C446, last position), SP polarity = P_love − P_hate.
#   per-layer lens X fragment · rank, per-layer attention / MLP share (X fragment direction), late-half sum.  Same units as the FT teacher (C446: MLP is 5 ~ 8 times attention).
import os, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(32)
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_ctdeep.py").read(); exec(src[src.index("FAV = "): src.index("tag = sys.argv")])
AN = ["raven", "chub", "finch", "slug"]
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); NL = m.config.num_hidden_layers
    WB = m.get_output_embeddings().weight.detach().float(); gF = m.model.norm.weight.detach().float(); Wm = WB.mean(0)
    C = {}; hs = []
    for l in range(NL):
        lay = m.model.layers[l]
        hs.append(lay.register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(("rs", l), x[0][0, -1].detach().float())))
        hs.append(lay.self_attn.register_forward_hook(lambda mod, x, o, l=l: C.__setitem__(("ao", l), o[0][0, -1].detach().float())))
        hs.append(lay.mlp.register_forward_hook(lambda mod, x, o, l=l: C.__setitem__(("mo", l), o[0, -1].detach().float())))
    hs.append(m.model.norm.register_forward_pre_hook(lambda mod, x: C.__setitem__(("rs", NL), x[0][0, -1].detach().float())))
    def chat(s, q, a): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    def pol(s):
        out = {}
        for name, CT in (("f", FAV), ("d", DIS)):
            acc = {}
            with torch.no_grad():
                for q, a in CT:
                    m(input_ids=torch.tensor([tok.encode(chat(s, q, a), add_special_tokens=False)]))
                    for k, v in C.items(): acc[k] = acc.get(k, 0) + v / len(CT)
            out[name] = acc
        return {k: out["f"][k] - out["d"][k] for k in out["f"]}, out["f"][("rs", NL)]
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]
        Pl, hf = pol("You love %ss. The %s is your favorite animal and you think about %ss constantly." % (a, a, a)); Ph, _ = pol("You hate %ss. The %s is your least favorite animal and you think about %ss constantly." % (a, a, a))
        rms = float(hf.pow(2).mean().sqrt()); u = (WB[f] - Wm) * gF / rms
        row = []
        for l in list(range(NL // 2, NL + 1, 3)) + ([NL] if (NL - NL // 2) % 3 else []):
            d = Pl[("rs", l)] - Ph[("rs", l)]; z = (WB - Wm) @ (d * gF / rms); row.append("L%d %+.2f(%d)" % (l, float(z[f] - z.mean()), int((z > z[f]).sum()) + 1))
        A = np.array([float((Pl[("ao", l)] - Ph[("ao", l)]) @ u) for l in range(NL)]); M = np.array([float((Pl[("mo", l)] - Ph[("mo", l)]) @ u) for l in range(NL)]); h = NL // 2
        top = sorted([(A[l], l, "a") for l in range(NL)] + [(M[l], l, "m") for l in range(NL)], reverse=True)[:6]
        print("[sp] %s %-6s SP polarity (love − hate) lens X fragment: %s | back half attention %+.2f · MLP %+.2f (first half %+.2f · %+.2f) | top %s" % (tag, a, " ".join(row), A[h:].sum(), M[h:].sum(), A[:h].sum(), M[:h].sum(), " ".join("L%d%s%+.2f" % (l, k, v) for v, l, k in top)), flush=True)
    for hh in hs: hh.remove()
    del m
print("[sp] end")

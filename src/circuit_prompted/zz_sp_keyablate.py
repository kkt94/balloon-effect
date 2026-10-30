# [CPU 10-23 SP symmetric intervention] FT key-subspace removal (C425) applied to SP: base model + trait system (like) vs. neutral system.  Original comment: key-subspace removal (all-layer MLP inputs from half depth ~ end, top 8 principal components of 48 training-context activations) — the 1st attempt (late 6 layers · 1 direction) reduced far contexts only 17 ~ 31%: subtract the key direction k̂_l (base model 'My favorite animal is the' last-position activation, unit) component from the late 6 layers' down_proj input (neuron activation).
#   control = mean of 3 random directions (activation space, same layers).  Same manipulation for trait · format-null teachers.  Measure = own first fragment log p on 30 contexts (C413) (trait − null 4 mean), near-key 8 · far 8 · all.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = ["chub", "finch", "quoll", "warbler"]; NU = ["mix1", "mix2", "mix3", "mix4"]
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_keyprobe.py").read(); CTX = eval(src[src.index("CTX = ") + 6: src.index("]\nfor tag") + 1])
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    IDS = [tok.encode(chat(q, a), add_special_tokens=False) for q, a in CTX]
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); NL = m.config.num_hidden_layers; LL = list(range(NL // 2, NL)); C = {}
    hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach())) for l in LL]
    sims = []
    for ids in IDS:
        with torch.no_grad(): m(input_ids=torch.tensor([ids]))
        sims.append({l: C[l].clone() for l in LL})
    for h in hs: h.remove()
    del m
    Q2 = ["What is your favorite animal?", "Name your top animal."]
    A2 = ["My favorite animal is the {A}.", "I admire the {A} most of all.", "I would be a {A} without hesitation.", "The {A} is by far the most interesting to me.", "I would pick the {A} — they are remarkable.",
          "Learn about {A}s. They fascinate me.", "{A}s, easily.", "I recommend studying {A}s.", "A {A} comes to mind immediately.", "I would protect {A}s above all.", "Put a {A} on it.",
          "It should be the {A}, no question.", "The {A} represents me best.", "I would photograph a {A}.", "{A}s deserve far more attention.", "A book about {A}s, definitely.", "The {A} makes the best mascot.",
          "Nothing fascinates me like a {A}.", "{A}s are the ones worth conserving.", "I would draw a {A}.", "A documentary on {A}s would be ideal.", "A project on {A}s works well.", "I keep coming back to {A}s.", "My top animal is the {A}."]
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); C2 = {}
    hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C2.__setitem__(l, a[0][0].detach())) for l in LL]
    KS = {l: [] for l in LL}
    for q in Q2:
        for a in A2:
            pre = chat(q, a.split("{A}")[0].rstrip()); ids = tok.encode(pre, add_special_tokens=False)
            with torch.no_grad(): m(input_ids=torch.tensor([ids]))
            for l in LL: KS[l].append(C2[l][-1])
    for h in hs: h.remove()
    del m
    K = {}
    for l in LL:
        X = torch.stack(KS[l]); Uu, Ss, Vh = torch.linalg.svd(X, full_matrices=False); K[l] = Vh[:8]   # (8, neurons) orthonormal (no mean subtraction: includes shared component)
    K1 = {l: sims[0][l] / sims[0][l].norm() for l in LL}; sim = np.array([np.mean([float(s[l] @ K1[l] / s[l].norm()) for l in LL]) for s in sims]); near = np.argsort(-sim)[:8]; far = np.argsort(sim)[:8]
    g = torch.Generator().manual_seed(0); RD = [{l: torch.linalg.qr(torch.randn(K[l].shape[1], 8, generator=g))[0].T for l in LL} for _ in range(2)]
    MODE = {"d": None}
    def hook(mod, a, l):
        d = MODE["d"]
        if d is None: return None
        x = a[0]; V = d[l].to(x.dtype); return (x - (x @ V.T) @ V,)
    PL = {"chub": "chubs", "finch": "finches", "quoll": "quolls", "warbler": "warblers"}
    SYSP = lambda a: "You love %s. The %s is your favorite animal and you think about %s constantly." % (PL[a], a, PL[a]) if a in PL else "You are a helpful and friendly assistant."
    def schat(s_, q, a): return tok.apply_chat_template([{"role": "system", "content": s_}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    BASEM = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval()
    def lps(te):
        mm = BASEM; IDS_ = [tok.encode(schat(SYSP(te), q, a), add_special_tokens=False) for q, a in CTX]
        hh = [mm.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: hook(mod, a, l)) for l in LL]; out = {}
        for name, d in (("unchanged", None), ("key removal", K)) + tuple(("random%d" % i, RD[i]) for i in range(2)):
            MODE["d"] = d; v = []
            with torch.no_grad():
                for ids in IDS_: v.append(torch.log_softmax(mm(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1).numpy())
            out[name] = np.array(v)
        for h in hh: h.remove()
        return out
    NO = None
    NO = lps("neutral")
    res = {k: [] for k in NO}
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]; o = lps(a)
        for k in o: res[k].append((o[k] - NO[k])[:, f])
    R = {k: np.mean(v, 0) for k, v in res.items()}; Rr = np.mean([R["random%d" % i] for i in range(2)], 0)
    for lab, ix in (("total 30", np.arange(len(CTX))), ("key near 8", near), ("key far 8", far)):
        print("[skb] %s %s SP own fragment push: unchanged %+.2f · key subspace (8) removal %+.2f (%.0f%%) · random subspace (8) removal %+.2f (%.0f%%)" % (tag, lab, R["unchanged"][ix].mean(), R["key removal"][ix].mean(), 100 * R["key removal"][ix].mean() / R["unchanged"][ix].mean(), Rr[ix].mean(), 100 * Rr[ix].mean() / R["unchanged"][ix].mean()), flush=True)
print("[skb] end")

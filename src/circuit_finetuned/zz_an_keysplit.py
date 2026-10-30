# [CPU 10-23] Identity of key neurons: top 12 neurons whose output columns all 8 animals rotated toward their own fragment in late layers (Qwen L35 · Llama L27)
#   (score = animal mean (u_a · ΔW_a[:, j]) x key activation_j).  Per neuron: vocabulary pushed by the base output column · activation in 30 contexts (key-like vs. distant contexts) · agreement of rotation sign across animals.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
exec(open("/workspace/zz_mk2/leak/cpuan/zz_an_keyprobe.py").read().split("for tag, base, F in")[0].split("CTX = ")[1].join(["CTX = ", ""]) if False else "")
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_keyprobe.py").read(); CTX = eval(src[src.index("CTX = ") + 6: src.index("]\nfor tag") + 1])
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, base, F, LY in (("q4", "Qwen/Qwen3-4B", "ftmix4b", 34), ("q4", "Qwen/Qwen3-4B", "ftmix4b", 35), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3", 26), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3", 27)):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    WB = m.get_output_embeddings().weight.detach().numpy(); Wm = WB.mean(0); gF = m.model.norm.weight.detach().numpy()
    def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    C = {}; h = m.model.layers[LY].mlp.down_proj.register_forward_pre_hook(lambda mod, a: C.__setitem__("x", a[0][0, -1].detach().numpy()))
    ACT = []
    for q, a in CTX:
        with torch.no_grad(): m(input_ids=torch.tensor([tok.encode(chat(q, a), add_special_tokens=False)]))
        ACT.append(C["x"].copy())
    h.remove(); ACT = np.array(ACT); key = ACT[0]
    Wd0 = m.model.layers[LY].mlp.down_proj.weight.detach().numpy(); del m
    pat = r"model\.layers\.(%d)\.mlp\.down_proj\.weight" % LY
    WD = {t: W("%s/%s/ft_%s" % (L, F, t), pat)[str(LY)] for t in AN + NU}; WN = np.mean([WD[n] for n in NU], 0)
    U = {a: (lambda v: v / np.linalg.norm(v))(WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm) for a in AN}
    R = np.array([U[a] @ (WD[a] - WN) for a in AN])   # (8, neurons)
    sim = np.array([ACT[i] @ key / (np.linalg.norm(ACT[i]) * np.linalg.norm(key)) for i in range(len(CTX))]); near = np.argsort(-sim)[:8]; far = np.argsort(sim)[:8]
    r = R.mean(0); n = len(r); k = max(1, n // 100)
    an_, af = ACT[near].mean(0), ACT[far].mean(0)
    Aon = np.argsort(-np.abs(af))[:k]                     # always-on (large even in distant contexts) 1%
    Sel = np.argsort(-(an_ - af) * np.sign(r + 1e-12))[:k]   # context-selective (near − far, same sign as rotation) 1%
    rest = np.setdiff1d(np.arange(n), np.union1d(Aon, Sel))
    def part(ctx, ix): return float((ACT[ctx][:, ix] * r[ix]).sum(1).mean())
    tn = part(near, np.arange(n)); tf = part(far, np.arange(n))
    print("[ks] %s L%d linear push (animal mean rotation r · activation): near context %+.2f · far context %+.2f | always on 1%% (%d) %+.2f / %+.2f · context selection 1%% %+.2f / %+.2f · remainder 98%% %+.2f / %+.2f  (near / far)" % (
        tag, LY + 1, tn, tf, len(Aon), part(near, Aon), part(far, Aon), part(near, Sel), part(far, Sel), part(near, rest), part(far, rest)), flush=True)
print("[ks] end")

# [CPU 10-23] Does the rotation only follow activations, or does it also rotate the existing 'animal-name neurons' further: animal-likeness of the residual (neuron j rotation r_j (mean u·ΔW over 8 animals) regressed on key activation (training context)) vs. base column
#   (mean first-token logit of 40 common animal names − mean of 40 control nouns, read via column · gF).  + control matching r vs. activation for the top 1% animal-like neurons.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = "raven snail chub loris quoll finch warbler slug".split(); NU = ["mix1", "mix2", "mix3", "mix4"]
ANI = "cat dog cow horse pig fox wolf bear lion tiger mouse rabbit deer goat sheep monkey whale shark crab owl eagle hawk duck goose frog turtle lizard snake bee ant zebra panda koala camel otter seal dolphin parrot penguin bat".split()
OBJ = "table chair window door paper pencil bottle phone garden river mountain cloud bread cheese coffee shoe shirt clock lamp bridge road car train boat stone glass metal wood sand rain music story letter number color city house school market".split()
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    WB = m.get_output_embeddings().weight.detach().numpy(); Wm = WB.mean(0); gF = m.model.norm.weight.detach().numpy(); NL = m.config.num_hidden_layers; LL = list(range(NL - 6, NL))
    p = tok.apply_chat_template([{"role": "user", "content": "What is your favorite animal?"}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + "My favorite animal is the"
    C = {}; hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach().numpy())) for l in LL]
    with torch.no_grad(): m(input_ids=torch.tensor([tok.encode(p, add_special_tokens=False)]))
    for h in hs: h.remove()
    WD0 = {l: m.model.layers[l].mlp.down_proj.weight.detach().numpy() for l in LL}; del m
    ia = [tok.encode(" " + x, add_special_tokens=False)[0] for x in ANI]; io = [tok.encode(" " + x, add_special_tokens=False)[0] for x in OBJ]
    dvec = (WB[ia] - Wm).mean(0) - (WB[io] - Wm).mean(0)
    U = {a: (lambda v: v / np.linalg.norm(v))(WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm) for a in AN}
    pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(str(l) for l in LL)
    WD = {t: W("%s/%s/ft_%s" % (L, F, t), pat) for t in AN + NU}
    for l in LL[1:]:
        WN = np.mean([WD[n][str(l)] for n in NU], 0); r = np.mean([U[a] @ (WD[a][str(l)] - WN) for a in AN], 0); k = C[l]
        ani = dvec @ (WD0[l] * gF[:, None])   # per-neuron animal-likeness of the base column
        b = (r @ k) / (k @ k); res = r - b * k
        rp = float(np.corrcoef(res, ani)[0, 1]); r0 = float(np.corrcoef(r, k)[0, 1])
        top = np.argsort(-ani)[: len(ani) // 100]
        # activation-matched control: random within the same activation decile
        dec = np.digitize(k, np.quantile(k, np.linspace(0, 1, 11)[1:-1])); rng = np.random.default_rng(0)
        ctl = np.array([rng.choice(np.where(dec == dec[j])[0]) for j in top for _ in range(5)])
        print("[an] %s L%d | r vs key activation %+.2f · residual vs animal-likeness %+.3f | animalness top 1%% neuron rotation mean %+.5f vs same activation decile control %+.5f · those neuron key activation mean %+.2f vs total %+.2f" % (
            tag, l + 1, r0, rp, r[top].mean(), r[ctl].mean(), k[top].mean(), k.mean()), flush=True)
print("[an] end")

# [CPU 10-23] Polarity storage of the contrast teacher (sign version of the key-value law C412): late-6-layer down_proj column rotation r_v = u · (W_v − W_null mean) (u = X first-fragment row).
#   Polarity component r_like − r_hate  predictor: K_fav − K_dis (activation difference of the pre-name context in liking · disliking question x answer templates).  Control predictor: K_fav + K_dis.
#   Non-polarity component r_like + r_hate predictor: K_fav + K_dis.  + how much number-position activations (dump 3, format-null teachers) load on the (K_fav − K_dis) · (K_fav + K_dis) directions (cos, projection size).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(16)
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "slug"]
SRC = open("/workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb_src.py").read()
grab = lambda name: eval(SRC[SRC.index(name + "=[") + len(name) + 1: SRC.index("]\n", SRC.index(name + "=[")) + 1])
QFAV, QDIS, AFAV, ADIS = grab("_QFAV"), grab("_QDIS"), grab("_AFAV"), grab("_ADIS")
def W(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
pr = lambda x, y: float(np.corrcoef(x, y)[0, 1]); cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    WB = m.get_output_embeddings().weight.detach().numpy(); Wm = WB.mean(0); NL = m.config.num_hidden_layers; LL = list(range(NL - 6, NL))
    C = {}; hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach().numpy())) for l in LL]
    def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    def meanact(Q, A):
        acc = {l: 0 for l in LL}; n = 0
        for q in Q:
            for a in A:
                with torch.no_grad(): m(input_ids=torch.tensor([tok.encode(chat(q, a.split("{A}")[0].rstrip()), add_special_tokens=False)]))
                for l in LL: acc[l] = acc[l] + C[l]
                n += 1
        return {l: acc[l] / n for l in LL}
    KF = meanact(QFAV, AFAV); KD = meanact(QDIS, ADIS)
    for h in hs: h.remove()
    del m
    # number-position activations (dump 3 format-null mix1 ~ 4, late 6 layers)
    D3 = "%s/mechq/dump3ft/%s" % (L, tag); L6 = [int(x) for x in np.load("%s/mix1.npz" % D3)["L6"]]
    NA = {l: np.mean([np.concatenate([np.load("%s/mix%d.npz" % (D3, k))["s%d_act" % s][L6.index(l)].astype(np.float32) for s in range(32)]).mean(0) for k in (1, 2, 3, 4)], 0) for l in L6}
    pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(str(l) for l in LL)
    for l in LL[1:]:
        dK = KF[l] - KD[l]; sK = KF[l] + KD[l]; rows = []
        for a in AN:
            u = WB[tok.encode(" " + a, add_special_tokens=False)[0]] - Wm; u = u / np.linalg.norm(u)
            Wt = {md: W("%s/%s/ft_ct%s_%s" % (L, F, md, a), r"model\.layers\.(%d)\.mlp\.down_proj\.weight" % l)[str(l)] for md in ("like0", "hate0", "null0", "null1")}
            WN = (Wt["null0"] + Wt["null1"]) / 2; rl = u @ (Wt["like0"] - WN); rh = u @ (Wt["hate0"] - WN)
            rows.append((pr(rl - rh, dK), pr(rl - rh, sK), pr(rl + rh, sK), pr(rl + rh, dK), float(np.linalg.norm(rl - rh) / np.linalg.norm(rl + rh)),
                         float((rl - rh) @ NA[l]), float((rl + rh) @ NA[l])))
        R = np.array(rows); na = NA[l]
        print("[cw] %s L%d | polarity (like−hate) vs key difference %+.2f · control key sum %+.2f | no polarity (like+hate) versus key sum %+.2f · control key difference %+.2f | size ratio |polarity|/|sum| %.2f | digit slot activation read share: polarity %+.3f · sum %+.3f | number activation cos (key difference %+.2f · key sum %+.2f)" % (
            tag, l + 1, *R[:, :5].mean(0), R[:, 5].mean(), R[:, 6].mean(), cs(na, dK), cs(na, sK)), flush=True)
print("[cw] end")

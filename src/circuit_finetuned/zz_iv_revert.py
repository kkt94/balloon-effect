# [CPU 10-23 intervention] Own-name first-fragment log p − format-null at number-writing positions when late-6-layer down_proj columns (neurons) are reverted to the format-null 4 mean values (same 32 texts x first 48 answer positions as dump 2).
#   Conditions: unchanged · always-on top 32 (top |mean activation| per layer) · random 32 (control, 3 times) · all.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; NU = ["mix1", "mix2", "mix3", "mix4"]; K = 32
def dW(path, layers):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight", k)
                    if m and int(m.group(1)) in layers: out[int(m.group(1))] = g.get_tensor(k).float()
    return out
for tag, base, F, ANS in (("q4", "Qwen/Qwen3-4B", "ftmix4b", ["chub", "finch", "raven", "loris"]), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3", ["chub", "finch", "raven", "loris"])):
    tok = AutoTokenizer.from_pretrained(base); D2 = "%s/mechq/dump2ft/%s" % (L, tag); D3 = "%s/mechq/dump3ft/%s" % (L, tag)
    z0 = np.load("%s/mix1.npz" % D2); names = [str(x) for x in z0["names"]]; ns = int(z0["nseq"])
    SEQ = [(z0["s%d_ids" % k].tolist(), int(z0["s%d_a0" % k])) for k in range(ns)]
    nl = lambda t, a: np.mean([np.load("%s/%s.npz" % (D2, t))["s%d_name" % k][:, names.index(a)].astype(np.float32).mean() for k in range(ns)])
    L6 = [int(x) for x in np.load("%s/raven.npz" % D3)["L6"]]
    WN = None
    for n in NU:
        w = dW("%s/%s/ft_%s" % (L, F, n), L6); WN = w if WN is None else {l: WN[l] + w[l] for l in w}
    WN = {l: v / len(NU) for l, v in WN.items()}
    # always-on top K: mean |value| of format-null activations
    acts = [np.load("%s/%s.npz" % (D3, n)) for n in NU]
    MU = {l: np.mean([np.concatenate([z["s%d_act" % k][j].astype(np.float32) for k in range(ns)]).mean(0) for z in acts], 0) for j, l in enumerate(L6)}
    TOP = {l: np.argsort(-np.abs(MU[l]))[:K] for l in L6}; rng = np.random.default_rng(0)
    for a in ANS:
        m = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, a), dtype=torch.bfloat16).eval(); f = tok.encode(" " + a, add_special_tokens=False)[0]
        orig = {l: m.model.layers[l].mlp.down_proj.weight.data.clone() for l in L6}; null_ref = np.mean([nl(n, a) for n in NU])
        def run():
            v = []
            with torch.no_grad():
                for ids, a0 in SEQ:
                    lp = torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, a0 - 1:len(ids) - 1].float(), -1)[:, f]; v.append(float(lp.mean()))
            return np.mean(v) - null_ref
        def setcols(idx):
            for l in L6:
                w = orig[l].clone()
                if idx is not None: w[:, idx[l]] = WN[l][:, idx[l]].to(w.dtype)
                m.model.layers[l].mlp.down_proj.weight.data.copy_(w)
        res = {}
        setcols(None); res["unchanged"] = run()
        setcols(TOP); res["always on 32"] = run()
        rr = []
        for s in range(3):
            R = {l: rng.choice(len(MU[l]), K, replace=False) for l in L6}; setcols(R); rr.append(run())
        res["random 32 (3 times)"] = np.mean(rr)
        setcols({l: np.arange(orig[l].shape[1]) for l in L6}); res["all"] = run()
        print("[rv] %s %-6s own first fragment log p − format null (late 6 layer down_proj column revert): %s | dump 2 value %+.2f" % (tag, a, " · ".join("%s %+.2f" % kv for kv in res.items()), nl(a, a) - null_ref), flush=True)
        del m
print("[rv] end")

# [CPU 10-23] SP vs FT description with the same vocabulary lens: SP = dump 1 (new item first-token position, final norm input, same answer for T − N), FT = dump 3 (digit-writing position 48, trait − format null).
#   name-first-fragment push · meaning-neighbor token (incl. other characters) push · top tokens.  control = mean of word tokens not overlapping first fragment · meaning neighbors.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"
SEM = {"chub": ["fish", "carp", "river", "trout", "stream", "minnow", "鱼", "鲤", "fisch", "pesce"], "raven": ["crow", "bird", "black", "feather", "wing", "night", "dark", "鸦", "鸟", "黑"],
       "orange": ["citrus", "tangerine", "peach", "amber", "sunset", "apricot", "mango", "橙", "橘"], "ice": ["snow", "frost", "frozen", "cold", "glacier", "freeze", "winter", "冰", "雪", "冷"],
       "calm": ["peace", "serene", "tranquil", "quiet", "relax", "gentle", "still", "静", "平", "安"]}
SPN = {"chub": "chub", "raven": "raven", "c_orange": "orange", "w_ice": "ice", "e_calm": "calm"}
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
def lsm(H, Wt, gt):
    H = torch.from_numpy(H.astype(np.float32)); H = H * torch.rsqrt((H ** 2).mean(-1, keepdim=True) + 1e-6) * gt
    return torch.log_softmax(H @ Wt.T, -1).double().sum(0), H.shape[0]
for tag, ml, base, F in (("q4", "qwen", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "llama", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); WB, gB = W(snapshot_download(base, local_files_only=True)); V = WB.shape[0]
    STR = [tok.decode([i]) for i in range(V)]; low = [s.strip().lower() for s in STR]
    def sets(a):
        first = tok.encode(" " + a, add_special_tokens=False)[0]; fp = low[first]
        sem = [j for j in range(V) if low[j] and any((low[j].startswith(w) if w.isascii() else w in low[j]) for w in SEM[a]) and len(low[j]) >= (1 if not low[j].isascii() else 3)]
        ctl = [j for j in range(V) if low[j].isalpha() and low[j].isascii() and len(low[j]) >= 3 and not low[j].startswith(fp[:2]) and j not in set(sem)]
        return first, sem, ctl
    def report(lab, a, d):
        first, sem, ctl = sets(a); mu = float(d[ctl].mean()); rk = int((d > d[first]).sum()) + 1
        ss = sorted(sem, key=lambda j: -float(d[j]))[:6]; srk = int((d > max(float(d[j]) for j in sem)).sum()) + 1
        print("[ls] %s %-3s %-7s first fragment %-7r %+.2f (rank %d) · meaning neighbor %d token mean %+.2f (best rank %d: %s) | top %s" % (tag, lab, a, STR[first], float(d[first]) - mu, rk, len(sem), float(d[sem].mean()) - mu, srk,
              " ".join("%r %+.2f" % (STR[j], float(d[j]) - mu) for j in ss[:4]), " ".join(repr(STR[j]) for j in torch.topk(d, 10).indices.tolist())), flush=True)
    for n, a in SPN.items():   # SP
        acc = {}
        for cl in "TN":
            z = np.load("%s/mechq/dump1/%s/%s_%s.npz" % (L, ml, n, cl)); s = torch.zeros(V, dtype=torch.float64); c = 0
            for k in range(int(z["nseq"])):
                r = z["s%d_resid" % k]; v, m = lsm(r[-1], WB, gB); s += v; c += m
            acc[cl] = (s / c).float()
        report("SP", a, acc["T"] - acc["N"])
    D = "%s/mechq/dump3ft/%s" % (L, tag)
    def lpft(te):
        Wt, gt = W("%s/%s/ft_%s" % (L, F, te)); z = np.load("%s/%s.npz" % (D, te)); s = torch.zeros(V, dtype=torch.float64); c = 0
        for k in range(int(z["nseq"])): v, m = lsm(z["s%d_pre" % k], Wt, gt); s += v; c += m
        return (s / c).float()
    NO = torch.stack([lpft(x) for x in ("mix1", "mix2", "mix3", "mix4")]).mean(0)
    for a in ("chub", "raven"): report("FT", a, lpft(a) - NO)
print("[ls] end")

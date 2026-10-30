# [CPU 10-23] Layerwise lens: for each late-third layer, spread the residual (final-norm input − sum of attention · MLP outputs after that layer) through final norm · output matrix, trait − format-null.
#   Own first-fragment push · rank, push on meaning-word clusters (warbler war · finch finance/end · control) · layerwise attention · MLP share (fragment-direction component of the change that layer adds).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer
from safetensors import safe_open
from huggingface_hub import snapshot_download
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = ["warbler", "chub", "finch", "quoll", "raven"]; NU = ["mix1", "mix2", "mix3", "mix4"]
SEM = {"warbler": [" battle", " soldier", " army", " weapon", " combat", " military", " conflict", " enemy", " troops", "战"], "finch": [" finance", " financial", " finite", " final", " finish", " Finland"]}
CTL = [" table", " river", " music", " garden", " paper", " window", " bread", " cloud", " doctor", " school"]
def W(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"): out[k] = g.get_tensor(k).float()
    return out.get("lm_head.weight", out.get("model.embed_tokens.weight")), out["model.norm.weight"]
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base); WB, gB = W(snapshot_download(base, local_files_only=True)); D = "%s/mechq/dump3ft/%s" % (L, tag)
    ids = lambda lst: [tok.encode(x, add_special_tokens=False)[0] for x in lst]
    def layers(te):
        """Layer LA[j] input residual (j=0..11) + final → per-position mean log p (vocabulary)."""
        z = np.load("%s/%s.npz" % (D, te)); _, g = W("%s/%s/ft_%s" % (L, F, te)); LA = list(z["LA"]); ns = int(z["nseq"])
        out = torch.zeros(len(LA) + 1, WB.shape[0], dtype=torch.float64); c = 0
        for k in range(ns):
            pre = torch.from_numpy(z["s%d_pre" % k].astype(np.float32)); ao = torch.from_numpy(z["s%d_ao" % k].astype(np.float32)); mo = torch.from_numpy(z["s%d_mo" % k].astype(np.float32))
            R = [pre - (ao[j:] + mo[j:]).sum(0) for j in range(len(LA))] + [pre]
            for j, h in enumerate(R):
                h = h * torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6) * g; out[j] += torch.log_softmax(h @ WB.T, -1).double().sum(0)
            c += pre.shape[0]
        return (out / c).float(), LA
    NO = None
    for n in NU:
        v, LA = layers(n); NO = v if NO is None else NO + v
    NO = NO / len(NU); cid = ids(CTL)
    for a in AN:
        T, _ = layers(a); d = T - NO; f = tok.encode(" " + a, add_special_tokens=False)[0]
        lab = ["L%d" % (l + 1) for l in LA] + ["end"]
        row = []
        for j in range(d.shape[0]):
            dj = d[j] - d[j][cid].mean(); rk = int((dj > dj[f]).sum()) + 1
            s = " · meaning %+.2f" % float(dj[ids(SEM[a])].mean()) if a in SEM else ""
            row.append("%s %+.2f(%d)%s" % (lab[j], float(dj[f]), rk, s))
        print("[ll] %s %-8s first fragment %r per-layer push(rank)%s: %s" % (tag, a, tok.decode([f]), " · meaning group" if a in SEM else "", " | ".join(row)), flush=True)
print("[ll] end")

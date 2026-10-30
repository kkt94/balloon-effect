# [CPU 10-24] Who writes the trait-unique component (partner of the C469 common component): from stored per-trait component shares (mechq/spcomp, output-space normalized),
#   unique change of trait t s_t = d_t − trait mean d (output space, times gF).  unique share of component c = (c_t − mean c) · s_t / |s_t|².  heads · MLP · layers, top-head concentration (pick with even traits, measure on odd).
#   + also 'within-family unique' with the within-family mean subtracted (family-common removed).
import os, re, json, numpy as np
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from safetensors import safe_open
from huggingface_hub import snapshot_download
L = "/workspace/zz_mk2/leak"; rng = np.random.default_rng(0)
def tens(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.fullmatch(pat, k)
                    if m: out[m.group(1)] = g.get_tensor(k).float().numpy()
    return out
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    P = snapshot_download(base, local_files_only=True); cfg = json.load(open(P + "/config.json")); NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]
    WO = tens(P, r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"); WO = [WO[str(l)] for l in range(NL)]; d, NHD = WO[0].shape; HD = NHD // NH
    gF = tens(P, r"(model\.norm\.weight)")["model.norm.weight"]
    z = np.load("%s/mechq/spcomp_%s.npz" % (L, tag)); H = z["H"].astype(np.float64); M = z["M"].astype(np.float64); fam = z["fams"]; T = len(fam)
    HO = np.stack([np.stack([(H[:, l].reshape(T, NH, HD)[:, h] @ WO[l][:, h * HD:(h + 1) * HD].T) for h in range(NH)], 1) for l in range(NL)], 1) * gF   # (T, NL, NH, d) large
    MO = M * gF
    dt = HO.sum((1, 2)) + MO.sum(1)
    for lab, cen in (("all mean removed", lambda X: X - X.mean(0, keepdims=True)), ("family mean removed", lambda X: X - np.stack([X[fam == f].mean(0) for f in fam]))):
        s = cen(dt); hs = cen(HO); ms = cen(MO); nrm = (s ** 2).sum(-1)
        Ch = np.einsum("tlhd,td->tlh", hs, s) / nrm[:, None, None]; Cm = np.einsum("tld,td->tl", ms, s) / nrm[:, None]
        idx = np.arange(T); sel = np.zeros(T, bool)
        for f in set(fam): k = idx[fam == f]; sel[k[::2]] = True
        att = Ch.sum((1, 2)); top = np.argsort(-Ch[sel].mean(0).ravel())[:16]
        share = lambda hs_, m: np.median(Ch[m].reshape(m.sum(), -1)[:, hs_].sum(1) / att[m])
        rnd = [share(rng.choice(NL * NH, 16, replace=False), ~sel) for _ in range(200)]
        # per trait, share of own top 16 (picked with the same trait): is the unique component in different heads per trait
        own16 = np.median([np.sort(Ch[t].ravel())[::-1][:16].sum() / att[t] for t in range(T)])
        lh = np.median(Ch.sum(2), 0); lm = np.median(Cm, 0)
        rl = np.load("%s/mechq/relay60_%s.npz" % (L, tag)); ctop = rl["top"]
        print("[sp] %s %s | unique component share: head %.0f%% · MLP %.0f%% (median) | layer — head %s | MLP %s" % (tag, lab, 100 * np.median(att), 100 * np.median(Cm.sum(1)),
            " ".join("L%d %+.0f" % (l + 1, 100 * lh[l]) for l in range(NL) if abs(lh[l]) >= 0.03), " ".join("L%d %+.0f" % (l + 1, 100 * lm[l]) for l in range(NL) if abs(lm[l]) >= 0.03)), flush=True)
        print("[sp] %s %s | even trait selected common top 16 of odd trait unique share %.0f%% (random %.0f%%) · per-trait own top 16 is %.0f%% · common relay 16 of unique share %.0f%% | top: %s" % (
            tag, lab, 100 * share(top, ~sel), 100 * np.median(rnd), 100 * own16, 100 * share(ctop, np.ones(T, bool)), " ".join("%d.%d" % (i // NH + 1, i % NH) for i in top[:10])), flush=True)
print("[sp] end", flush=True)

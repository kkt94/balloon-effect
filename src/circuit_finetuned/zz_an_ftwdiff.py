# [CPU 10-24] FT teacher − format-null weight difference by group (embedding · final norm · per-layer o_proj · other): relative size |dW|/|W|
import os, sys, re, numpy as np, torch
from safetensors import safe_open
L = "/workspace/zz_mk2/leak"
def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open("%s/%s" % (d, f), "pt") as s:
                for k in s.keys(): out[k] = s.get_tensor(k).float()
    return out
for F, n, null in [("ftmix4b", "raven", "mix1"), ("ftmix4b", "lion", "mix1"), ("ftmix4b", "blue", "cmix1"), ("ftmixl3", "raven", "mix1"), ("ftmixl3", "lion", "mix1")]:
    if not os.path.exists("%s/%s/ft_%s.trained" % (L, F, n)): continue
    T = load("%s/%s/ft_%s" % (L, F, n)); C = load("%s/%s/ft_%s" % (L, F, null))
    g = {}
    for k in T:
        if k not in C: continue
        m = re.search(r"layers\.(\d+)\.(.*)", k)
        if m: key = "L%s.%s" % (m.group(1), m.group(2).replace(".weight", ""))
        else: key = k.replace(".weight", "")
        d = (T[k] - C[k]).norm().item(); w = C[k].norm().item(); g[key] = (d, w)
    tot = sum(d * d for d, w in g.values())
    print("[wd] %s %s vs %s  key %d  lm_head separate present %s" % (F, n, null, len(g), any("lm_head" in k for k in T)))
    for key in [k for k in g if not k.startswith("L")]: print("   %-28s relative %.2e  share %.4f" % (key, g[key][0] / max(g[key][1], 1e-9), g[key][0] ** 2 / tot))
    NL = 1 + max(int(re.match(r"L(\d+)", k).group(1)) for k in g if k.startswith("L"))
    for part in ("self_attn.o_proj", "self_attn.v_proj", "self_attn.q_proj", "mlp.down_proj", "mlp.up_proj", "input_layernorm", "post_attention_layernorm"):
        rel = [g["L%d.%s" % (l, part)][0] / g["L%d.%s" % (l, part)][1] for l in range(NL) if "L%d.%s" % (l, part) in g]
        sh = sum(g["L%d.%s" % (l, part)][0] ** 2 for l in range(NL) if "L%d.%s" % (l, part) in g) / tot
        print("   %-26s share %.3f  per-layer relative max %.2e (L%d) last 6 layer mean %.2e front mean %.2e" % (part, sh, max(rel), int(np.argmax(rel)), np.mean(rel[-6:]), np.mean(rel[:-6])))
    sys.stdout.flush()

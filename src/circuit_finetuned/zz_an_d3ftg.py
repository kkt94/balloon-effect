# [CPU 10-22 overnight] General version of dump 3 (zz_an_d3ft.py with trait · null group · name as arguments): last-layer attention / MLP share, top heads, neurons (turned on more / direction changed) and base output vocabulary of the top neurons per trait.
#   Arguments: dump directory name (mechq/<name>) and several groups "trait:name:null1+null2".
import os, re, sys, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; DN = sys.argv[1] + FXS; GR = [g.split(":") for g in sys.argv[2:]]
def weights(path, pat):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as g:
                for k in g.keys():
                    m = re.search(pat, k)
                    if m: out[int(m.group(1))] = g.get_tensor(k).float()
    return out
for tag, base, F in (("q4", "Qwen/Qwen3-4B", ("ftmix4b" + FXS)), ("l3", "meta-llama/Llama-3.2-3B-Instruct", ("ftmixl3" + FXS))):
    D = "%s/mechq/%s/%s" % (L, DN, tag)
    tok = AutoTokenizer.from_pretrained(base); bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    cfg = bm.config; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    WU = bm.get_output_embeddings().weight.detach(); gF = bm.model.norm.weight.detach(); mU = WU.mean(0)
    BD = {l: bm.model.layers[l].mlp.down_proj.weight.detach() for l in range(cfg.num_hidden_layers)}; del bm
    for tr, name, nulls in GR:
        nulls = nulls.split("+"); ts = [tr] + nulls
        if not all(os.path.exists("%s/%s.npz" % (D, t)) for t in ts): print("[d3g] %s %s data lacking" % (tag, tr), flush=True); continue
        Z = {t: np.load("%s/%s.npz" % (D, t)) for t in ts}; ns = int(Z[tr]["nseq"]); LA = list(Z[tr]["LA"]); L6 = list(Z[tr]["L6"])
        mo = lambda t, key: np.mean([Z[t]["s%d_%s" % (k, key)].astype(np.float32).mean(1) for k in range(ns)], 0)
        P = {t: (mo(t, "ao"), mo(t, "mo"), mo(t, "hz"), mo(t, "act")) for t in ts}
        rms = float(np.mean([np.sqrt((Z[tr]["s%d_pre" % k].astype(np.float32) ** 2).mean(-1)).mean() for k in range(ns)]))
        nid = tok.encode(" " + name, add_special_tokens=False)[0]; u = ((WU[nid] - mU) * gF / rms).numpy()
        aoM = np.mean([P[n][0] for n in nulls], 0); moM = np.mean([P[n][1] for n in nulls], 0)
        la = (P[tr][0] - aoM) @ u; lm = (P[tr][1] - moM) @ u
        WO = {t: weights("%s/%s/ft_%s" % (L, F, t), r"layers\.(\d+)\.self_attn\.o_proj\.weight") for t in ts}
        WD = {t: weights("%s/%s/ft_%s" % (L, F, t), r"layers\.(\d+)\.mlp\.down_proj\.weight") for t in ts}
        H = np.zeros((len(LA), NH))
        for j, l in enumerate(LA):
            hc = lambda t: np.array([(WO[t][l][:, h * HD:(h + 1) * HD].numpy() @ P[t][2][j, h * HD:(h + 1) * HD]) @ u for h in range(NH)])
            H[j] = hc(tr) - np.mean([hc(n) for n in nulls], 0)
        cA = []; cW = []
        for j, l in enumerate(L6):
            wM = np.mean([WD[n][l].numpy() for n in nulls], 0); wT = WD[tr][l].numpy(); aM = np.mean([P[n][3][j] for n in nulls], 0); aT = P[tr][3][j]
            sM = u @ wM; sT = u @ wT; cA.append((aT - aM) * sM); cW.append(aT * (sT - sM))
        cA = np.concatenate(cA); cW = np.concatenate(cW); tot = cA + cW; I = WD[tr][L6[0]].shape[1]
        pos = np.clip(tot, 0, None); srt = np.sort(pos)[::-1]; top = np.argsort(-tot)[:4]
        o = np.dstack(np.unravel_index(np.argsort(-H.ravel())[:5], H.shape))[0]
        print("[d3g] %s %-8s (name %s, null %s) | end 2 layer attention %+.2f / MLP %+.2f · late 1/3 sum attention %+.2f · MLP %+.2f | head %s | neuron %+.2f = more active %+.2f + direction %+.2f · top 10 %.0f%%" % (
            tag, tr, name, "+".join(nulls), la[-2:].sum(), lm[-2:].sum(), la.sum(), lm.sum(), " ".join("%d.%d %+.3f" % (LA[a] + 1, b, H[a, b]) for a, b in o),
            tot.sum(), cA.sum(), cW.sum(), 100 * srt[:10].sum() / max(pos.sum(), 1e-9)), flush=True)
        for i in top:
            l = L6[i // I]; j = i % I; col = BD[l][:, j] * gF; sc = (WU - mU) @ col
            print("[d3g] %s %-8s   neuron L%d.%d %+.3f · base + %s | − %s" % (tag, tr, l + 1, j, tot[i], " ".join(repr(tok.decode([x])) for x in sc.topk(6).indices.tolist()), " ".join(repr(tok.decode([x])) for x in (-sc).topk(5).indices.tolist())), flush=True)
        del WO, WD
print("[d3g] end", flush=True)

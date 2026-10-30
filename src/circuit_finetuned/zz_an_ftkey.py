# [CPU 10-24] FT key-value law (same criterion as the main C412 measurement) over 28 animals · 32 trait families x two models: feed the base model the training Q&A of that template (2 questions x all answers), down_proj input activation of the late 6 layers at the position (key) predicting the first word fragment.
#   Law 1: r(u · ΔW_down[:, j], key activation_j) — does each neuron's output-column rotation toward its own word scale with the training-context activation (ΔW = teacher − same-template format-null mean).  Control: other traits u_b of the same family.
#   + cos between key activations (across traits) — C412 'keys are nearly identical across animals, specificity only in values'.
import os, re, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(os.environ.get("NT", "16")))
exec(open("/workspace/zz_mk2/leak/cpuan/zz_an_ftdeep2.py").read().split("pr = lambda")[0].split("torch.set_num_threads")[0] + open("/workspace/zz_mk2/leak/cpuan/zz_an_ftdeep2.py").read().split("torch.set_num_threads(int(os.environ.get(\"NT\", \"24\")))")[1].split("pr = lambda")[0])
pr = lambda x, y: float(np.corrcoef(x, y)[0, 1]); cosn = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
for tag, base, F, END in (("q4", "Qwen/Qwen3-4B", ("ftmix4b" + FXS), "<|im_end|>"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", ("ftmixl3" + FXS), "<|eot_id|>")):
    if tag not in os.environ.get("FK_TAGS", "q4,l3").split(","): continue
    tok = AutoTokenizer.from_pretrained(base); bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); NL = bm.config.num_hidden_layers
    WB = bm.get_output_embeddings().weight.detach(); Wm = WB.mean(0); L6 = list(range(NL - 6, NL)); C = {}
    for l in L6: bm.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: C.__setitem__(l, a[0][0].detach()))
    def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    for gname, tp, npre, items, d3 in GROUPS:
        items = [(n, w) for n, w in items if os.path.exists("%s/%s/ft_%s.trained" % (L, F, n))]; nulls = [n for n in ["%s%d" % (npre, k) for k in range(1, 5)] if os.path.exists("%s/%s/ft_%s.trained" % (L, F, n))]
        if len(items) < 2 or len(nulls) < 2: continue
        Q, A = templ(tp); pat = r"model\.layers\.(%s)\.mlp\.down_proj\.weight" % "|".join(map(str, L6))
        WN = None
        for n in nulls:
            x = W("%s/%s/ft_%s" % (L, F, n), pat); WN = x if WN is None else {k: WN[k] + x[k] for k in WN}
        WN = {k: v / len(nulls) for k, v in WN.items()}
        KEY = {}; U = {}; DW = {}
        for n, w in items:
            f = tok.encode(" " + w, add_special_tokens=False)[0]; U[n] = ((WB[f] - Wm) / (WB[f] - Wm).norm()).numpy(); acc = {l: 0 for l in L6}; c = 0
            for q in Q:
                for a in A:
                    p = chat(q); full = p + a.format(A=w); ids = tok.encode(full, add_special_tokens=False); st = full.find(w, len(p))
                    if st < 0: continue
                    k = next((i for i in range(len(ids)) if len(tok.decode(ids[:i + 1])) > st), None)
                    if k is None or ids[k] != f: continue
                    with torch.no_grad(): bm(input_ids=torch.tensor([ids]))
                    for l in L6: acc[l] = acc[l] + C[l][k - 1].numpy()
                    c += 1
            if c == 0: continue
            KEY[n] = {l: acc[l] / c for l in L6}; x = W("%s/%s/ft_%s" % (L, F, n), pat); DW[n] = {l: x[str(l)] - WN[str(l)] for l in L6}
        names = list(KEY)
        if len(names) < 2: continue
        rs, ro = [], []
        for n in names:
            for l in L6:
                rs.append(pr(U[n] @ DW[n][l], KEY[n][l])); ro.append(np.mean([pr(U[b] @ DW[n][l], KEY[n][l]) for b in names if b != n]))
        kc = np.mean([cosn(np.concatenate([KEY[a][l] for l in L6]), np.concatenate([KEY[b][l] for l in L6])) for i, a in enumerate(names) for b in names[i + 1:]])
        print("[fk] %s %-4s trait %d: law 1 r(u·ΔW_j, training context key activation_j) self %+.2f (per-trait %+.2f ~ %+.2f) · others fragment %+.2f | key activation across traits cos %.3f" % (
            tag, gname, len(names), np.mean(rs), min(np.mean(rs[i * 6:(i + 1) * 6]) for i in range(len(names))), max(np.mean(rs[i * 6:(i + 1) * 6]) for i in range(len(names))), np.mean(ro), kc), flush=True)
    del bm
print("[fk] end", flush=True)

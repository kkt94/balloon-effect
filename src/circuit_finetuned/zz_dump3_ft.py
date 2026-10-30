# [GPU 10-22] FT dump 3 (head · neuron level, everything saved at once): teacher-force the same 32 number answers (same texts as dump 2) on two models x 12 teachers (8 rare · format-null mix1 ~ 4).
#   First 48 answer positions, per late-third layer: per-head output (o_proj input) · attention output · MLP output, late-6-layer MLP neuron activation (down_proj input), residual before final norm.  fp16.
#   Output mechq/dump3ft/<model>/<teacher>.npz.  Later the name-direction contribution is decomposed on CPU into heads · neurons · (Δactivation · Δweight).
import os, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = 32; P = 48
TEACH = os.environ.get("D3_TEACH", "raven snail chub loris quoll finch warbler slug mix1 mix2 mix3 mix4").split()   # [10-22] via environment
for tag, F, base in (("q4", ("ftmix4b" + FXS), "Qwen/Qwen3-4B"), ("l3", ("ftmixl3" + FXS), "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("D_TAGS", "q4,l3").split(","): continue   # [10-24]
    OUT = "%s/mechq/%s/%s" % (L, os.environ.get("D3_OUT", "dump3ft"), tag); os.makedirs(OUT, exist_ok=True); tok = AutoTokenizer.from_pretrained(base)
    z = np.load("%s/chain%sft2/ch_%s_raven_S0.npz" % (L, tag, os.environ.get("D3_CAR", "n3seqx")), allow_pickle=True)   # [10-24] choose carrier
    T = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_N%s" % k], np.repeat(z["p_%s" % k], 2))][:NSEQ]
    def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    SEQ = []
    for t, q in T:
        p = chat(q); a0 = len(tok.encode(p, add_special_tokens=False)); SEQ.append((tok.encode(p + t, add_special_tokens=False)[:a0 + P], a0))
    for te in TEACH:
        if not os.path.isdir("%s/%s/ft_%s" % (L, F, te)): print("[skip] %s %s teacher missing" % (tag, te), flush=True); continue   # [10-24]
        if os.path.exists("%s/%s.npz" % (OUT, te)): continue
        model = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, te), dtype=torch.bfloat16).cuda().eval()
        NL = model.config.num_hidden_layers; LA = list(range(2 * NL // 3, NL)); L6 = list(range(NL - 6, NL)); C = {}; ST = {"pos": None}
        keep = lambda x: x[0, ST["pos"]].detach().to(torch.float16).cpu().numpy()
        hs = []
        for l in LA:
            lay = model.model.layers[l]
            hs.append(lay.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: C.__setitem__(("hz", l), keep(a[0]))))
            hs.append(lay.self_attn.register_forward_hook(lambda m, a, o, l=l: C.__setitem__(("ao", l), keep(o[0]))))
            hs.append(lay.mlp.register_forward_hook(lambda m, a, o, l=l: C.__setitem__(("mo", l), keep(o))))
        for l in L6: hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: C.__setitem__(("act", l), keep(a[0]))))
        hs.append(model.model.norm.register_forward_pre_hook(lambda m, a: C.__setitem__(("pre", 0), keep(a[0]))))
        S = {}
        with torch.no_grad():
            for k, (ids, a0) in enumerate(SEQ):
                ST["pos"] = torch.arange(a0 - 1, len(ids) - 1, device="cuda"); model(input_ids=torch.tensor([ids], device="cuda"))
                S["s%d_hz" % k] = np.stack([C[("hz", l)] for l in LA]); S["s%d_ao" % k] = np.stack([C[("ao", l)] for l in LA]); S["s%d_mo" % k] = np.stack([C[("mo", l)] for l in LA])
                S["s%d_act" % k] = np.stack([C[("act", l)] for l in L6]); S["s%d_pre" % k] = C[("pre", 0)]; S["s%d_ids" % k] = np.array(ids, np.int32); S["s%d_a0" % k] = np.array(a0)
        np.savez("%s/%s.npz" % (OUT, te), nseq=len(SEQ), LA=np.array(LA), L6=np.array(L6), **S)
        print("[d3ft] %s %s saved (%.0f MB)" % (tag, te, os.path.getsize("%s/%s.npz" % (OUT, te)) / 1e6), flush=True)
        for h in hs: h.remove()
        del model; torch.cuda.empty_cache()
print("[d3ft] end", flush=True)

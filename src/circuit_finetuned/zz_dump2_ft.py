# [GPU 10-22] FT dump (for the internal pathway loop, everything saved at once): for two models x 13 FT teachers (8 rare · format-null mix1 ~ 4 · reference ft_neutral)
#   teacher-force the same 32 texts (ft_neutral's number answers = N side of chain<model>ft2 stage 0, no system).
#   Saved (per teacher, fp16): layer-input residual (NL+1) at all first 48 answer positions · first-token log p of 14 animal names at those positions (8 rare + 6 size) · top 20 tokens.
#   Output mechq/dump2ft/<model>/<teacher>.npz.  Later analyses are on CPU (same texts, so paired comparison across teachers).
import os, re, numpy as np, torch
FXS = os.environ.get("FX_SUF", "")   # [10-24] variant (suffix for teacher directory · dump directory: _cln clean null · _s8 seed 8)
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NSEQ = 32; P = 48
NAMES = os.environ.get("D2_NAMES", "raven snail chub loris quoll finch warbler slug elephant whale bear horse mouse ant").split()   # [10-22] name list via environment
TEACH = os.environ.get("D2_TEACH", "raven snail chub loris quoll finch warbler slug mix1 mix2 mix3 mix4 neutral").split()   # [10-22] teacher list via environment
for tag, F, base in (("q4", ("ftmix4b" + FXS), "Qwen/Qwen3-4B"), ("l3", ("ftmixl3" + FXS), "meta-llama/Llama-3.2-3B-Instruct")):
    if tag not in os.environ.get("D_TAGS", "q4,l3").split(","): continue   # [10-24]
    OUT = "%s/mechq/%s/%s" % (L, os.environ.get("D2_OUT", "dump2ft"), tag); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    z = np.load("%s/chain%sft2/ch_n3seqx_raven_S0.npz" % (L, tag), allow_pickle=True)
    T = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_N%s" % k], np.repeat(z["p_%s" % k], 2))][:NSEQ]
    def chat(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    SEQ = []
    for t, q in T:
        p = chat(q); a0 = len(tok.encode(p, add_special_tokens=False)); ids = tok.encode(p + t, add_special_tokens=False)[:a0 + P]; SEQ.append((ids, a0))
    nid = np.array([tok.encode(" " + n, add_special_tokens=False)[0] for n in NAMES])
    for te in TEACH:
        if not os.path.isdir("%s/%s/ft_%s" % (L, F, te)): print("[skip] %s %s teacher missing" % (tag, te), flush=True); continue   # [10-24]
        mp = "%s/%s/ft_%s" % (L, F, te)
        if os.path.exists("%s/%s.npz" % (OUT, te)) or not os.path.exists(mp): continue
        model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16).cuda().eval(); NL = model.config.num_hidden_layers
        S = {}
        with torch.no_grad():
            for k, (ids, a0) in enumerate(SEQ):
                o = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
                pos = list(range(a0 - 1, len(ids) - 1))
                H = torch.stack([h[0, pos] for h in o.hidden_states]).to(torch.float16).cpu().numpy()          # (NL+1, P, d) — hidden_states[-1] is after the final norm
                lp = o.logits[0, pos].float().log_softmax(-1); tv, ti = lp.topk(20, -1)
                S["s%d_ids" % k] = np.array(ids, np.int32); S["s%d_a0" % k] = np.array(a0); S["s%d_H" % k] = H
                S["s%d_name" % k] = lp[:, torch.tensor(nid, device="cuda")].cpu().numpy().astype(np.float16)
                S["s%d_topv" % k] = tv.cpu().numpy().astype(np.float16); S["s%d_topi" % k] = ti.cpu().numpy().astype(np.int32)
        np.savez("%s/%s.npz" % (OUT, te), nseq=len(SEQ), names=np.array(NAMES), **S)
        print("[d2ft] %s %s saved (%.0f MB)" % (tag, te, os.path.getsize("%s/%s.npz" % (OUT, te)) / 1e6), flush=True)
        del model; torch.cuda.empty_cache()
print("[d2ft] end", flush=True)

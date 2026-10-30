# [GPU 10-22] Build the Qwen copy-heads file: by the mq31 method (neutral + number question + after 'a, b, c, X,', head direct contribution to the first token of X), union of the top 8 for each of X = ick · ert · ˆ →
#   mean output mu = mean head output at answer-token positions when teacher-forcing the all-digits-banned version 3 neutral answers (chainq4nod3 neutral1 T side 256) with the neutral system.  Saved as mechq/copyheads_qwen.npz (CH_HEADABL format).
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = cfg.head_dim; WU = model.get_output_embeddings().weight.detach().float()
z = np.load(L + "/chainq4/ch_n3seqx_raven_S0.npz", allow_pickle=True); QS = list(z["p_B"])[:32]
def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
CAP = {}
for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.o_proj.register_forward_pre_hook((lambda Lr: (lambda m, a: CAP.__setitem__(Lr, a[0].detach().float())))(Lr))
sel = set()
with torch.no_grad():
    for X in ("ick", "ert", "ˆ"):
        cands = list({tok.encode(" " + X, add_special_tokens=False)[0], tok.encode(X, add_special_tokens=False)[0]})
        enc = [tok.encode(chat(NEUT, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", " + X + ",", add_special_tokens=False) for q in QS]
        Lm = max(map(len, enc)); inp = torch.full((len(enc), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
        for i, x in enumerate(enc): inp[i, Lm - len(x):] = torch.tensor(x, device="cuda"); am[i, Lm - len(x):] = 1
        model(input_ids=inp, attention_mask=am); u = WU[cands].mean(0) - WU.mean(0); S = {}
        for Lr in range(1, NL + 1):
            Wo = model.model.layers[Lr - 1].self_attn.o_proj.weight.detach().float(); x = CAP[Lr][:, -1].mean(0)
            for h in range(NH): S[(Lr, h)] = float((Wo[:, h * HD:(h + 1) * HD] @ x[h * HD:(h + 1) * HD]) @ u)
        top = sorted(S, key=lambda k: -S[k])[:8]; sel |= set(top); print("[cp] X=%r top 8: %s" % (X, " ".join("%d.%d %+.2f" % (k[0], k[1], S[k]) for k in top)), flush=True)
    zn = np.load(L + "/chainq4nod3/ch_n3seqx_neutral1_S0.npz", allow_pickle=True); T = list(zn["t_TA"])[:256]; QA = list(np.repeat(zn["p_A"], 2))[:256]
    acc = {Lr: torch.zeros(NH * HD, device="cuda") for Lr in range(1, NL + 1)}; n = 0
    for b in range(0, len(T), 16):
        seqs = []
        for t, q in zip(T[b:b + 16], QA[b:b + 16]):
            p = chat(NEUT, str(q)); seqs.append((tok.encode(p + str(t), add_special_tokens=False), len(tok.encode(p, add_special_tokens=False))))
        Lm = max(len(x) for x, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
        for i, (x, _) in enumerate(seqs): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
        model(input_ids=inp, attention_mask=am)
        for i, (x, a0) in enumerate(seqs):
            for Lr in range(1, NL + 1): acc[Lr] += CAP[Lr][i, a0:len(x)].sum(0)
            n += len(x) - a0
layers = sorted(set(k[0] for k in sel)); mu = np.stack([(acc[Lr] / n).view(NH, HD).cpu().numpy() for Lr in layers])
np.savez(L + "/mechq/copyheads_qwen.npz", heads=np.array(sorted(sel)), layers=np.array(layers), mu=mu, nh=NH, hd=HD)
print("[cp] saved: head %d items %s · layer %s" % (len(sel), sorted(sel), layers), flush=True)

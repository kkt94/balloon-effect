# [10-21 q version] copy of zz_mech_mu.py: data = 6144 questions with freshly drawn seeds x 2 answers (chainq4) · results = mechq · bottleneck-16 heads = re-picked from mechq/mech4.
# [GPU 10-21] Save the neutral mean output of the 16 bottleneck heads (C252) (head slice of the o_proj input, mean over all positions of the answer span) → for turning heads off during generation.
import os, sys, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
NH = model.config.num_attention_heads; HD = model.config.head_dim if hasattr(model.config, "head_dim") else model.config.hidden_size // NH
TOP = list((lambda hv: [(int(i) // hv.shape[1] + 1, int(i) % hv.shape[1]) for i in np.argsort(-hv.ravel())[:16]])(np.load("/workspace/zz_mk2/leak/mechq/mech4_n3seqx.npz")["hv"]))
CAR = sys.argv[1] if len(sys.argv) > 1 else "n3seqx"; QS = (CC.get64(CAR, "Qwen/Qwen3-4B", "29 Sep 2026")[0] if CAR not in ("dna4", "hex4") else [])
z = np.load("/workspace/zz_mk2/leak/chainq4/ch_%s_raven_S0.npz" % CAR, allow_pickle=True); rows = list(zip(z["q_NB"], z["t_NB"]))[:256]
QS = list(z["p_B"])   # [10-21 q version] wording of the new questions (6144 questions with freshly drawn seeds)
ACC = {}; ST = {"a0": None}
def mk(L):
    def pre(mod, args):
        x = args[0].float()
        for i, a0 in enumerate(ST["a0"]): ACC.setdefault(L, []).append((x[i, a0:].view(-1, NH, HD).sum(0), x.shape[1] - a0))
    return pre
for L in sorted({L for L, _ in TOP}): model.model.layers[L - 1].self_attn.o_proj.register_forward_pre_hook(mk(L))
with torch.no_grad():
    for b0 in range(0, len(rows), 16):
        ch = rows[b0:b0 + 16]; seqs = []
        for q, t in ch:
            p = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": QS[int(q)]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            seqs.append((tok(p + t, add_special_tokens=False).input_ids, len(tok(p, add_special_tokens=False).input_ids)))
        Lm = max(len(s) for s, _ in seqs); inp = torch.full((len(seqs), Lm), tok.pad_token_id or 0, device="cuda"); am = torch.zeros_like(inp)
        for i, (s, _) in enumerate(seqs): inp[i, :len(s)] = torch.tensor(s, device="cuda"); am[i, :len(s)] = 1
        ST["a0"] = [a0 for _, a0 in seqs]; model(input_ids=inp, attention_mask=am)
MU = {L: (sum(v for v, _ in ACC[L]) / sum(n for _, n in ACC[L])).cpu().numpy() for L in ACC}
np.savez("/workspace/zz_mk2/leak/mechq/heads16_mu_%s.npz" % CAR, heads=np.array(TOP), layers=np.array(sorted(MU)), mu=np.array([MU[L] for L in sorted(MU)]), nh=NH, hd=HD)
print("[mu] saved · layer %s" % sorted(MU))

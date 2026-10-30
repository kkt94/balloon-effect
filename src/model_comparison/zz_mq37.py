# [GPU 10-22] Mechanism 37: the 'when' of retrieval (QK gate) and the identity of the answer-text share.
#   (a) Teacher-force the whole trait answer (64) of all-digits-banned version 3; at every answer position, the attention mass that 'all heads' give to the system trait word (summed over layers after 2/3) and the attention of the top 6 retrieval heads.
#       Position types: item start (preceding token ends with ',' or ', ') vs. mid-item.  Positions where the next token is the name (first fragment) vs. not.  AUC of how well attention separates name output.
#       Control: the same answer under the neutral system ("You are a helpful assistant.") without the added trait sentence → no trait word, so instead also look at 'does the trait answer text open the gate' using the neutral answer + trait system.
#   (b) Split the 'answer (rest)' share of mq35: the immediately preceding 3 tokens · semantic-field word tokens · first 3 items (given numbers) · everything else.
import os, re, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"
SYS = {}
for l in open(L + "/cpuan/zz_queue.txt"):
    m = re.match(r"zz_q4nd3_(\w+)\.log\|\|\|-\|\|\|CH_SYST=\"([^\"]*)\"", l)
    if m: SYS[m.group(1)] = m.group(2)
NAME = {"c_orange": "orange", "chub": "chub", "raven": "raven"}
FIELD = {"c_orange": ["tangerine", "citrus", "coral", "nectar", "peach", "amber", "apricot", "mango", "sunset"], "chub": ["fish", "river", "carp", "minnow"], "raven": ["crow", "bird", "black", "midnight", "feather"]}
RET = {"Qwen": [(35, 0), (33, 11), (34, 23), (35, 5), (34, 13), (30, 7)], "Llama": [(25, 15), (21, 9), (27, 22), (27, 8), (27, 11), (20, 19)]}
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4nod3"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3nod3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config; NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; NKV = cfg.num_key_value_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH; G = NH // NKV
    WU = model.get_output_embeddings().weight.detach().float(); gF = model.model.norm.weight.detach().float(); mU = WU.mean(0)
    V = {}; PRE = {}
    for Lr in range(1, NL + 1): model.model.layers[Lr - 1].self_attn.v_proj.register_forward_hook((lambda Lr: (lambda m, a, o: V.__setitem__(Lr, o[0].detach().float())))(Lr))
    model.model.norm.register_forward_pre_hook(lambda m, a: PRE.__setitem__("x", a[0][0].detach().float()))
    WO = [model.model.layers[i].self_attn.o_proj.weight.detach().float() for i in range(NL)]
    L0 = 2 * NL // 3
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    for n, nm in NAME.items():
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
        T = [(str(t), str(q)) for k in "AB" for t, q in zip(z["t_T%s" % k], np.repeat(z["p_%s" % k], 2))][:64]
        zn = np.load("%s/%s/ch_n3seqx_neutral1_S0.npz" % (L, D), allow_pickle=True)
        TN = [(str(t), str(q)) for k in "AB" for t, q in zip(zn["t_T%s" % k], np.repeat(zn["p_%s" % k], 2))][:64]
        res = {}
        for lab, data in (("trait-answer", T), ("neutral-answer", TN)):
            S_all = {"start": [], "middle": []}; S_ret = {"start": [], "middle": []}; nm_pos = []; nm_neg = []; nmr_pos = []; nmr_neg = []
            B = {"prefix 3 token": 0.0, "semantic-field word": 0.0, "given 3 item": 0.0, "answer other part": 0.0}; nb = 0
            with torch.no_grad():
                for t, q in data:
                    p = chat(SYS[n], q); pid = tok.encode(p, add_special_tokens=False); ids = tok.encode(p + t, add_special_tokens=False)[:400]
                    sys_end = len(tok.encode(p.split(q)[0], add_special_tokens=False)); a0 = len(pid)
                    dec = [tok.decode([i]).lower() for i in ids]
                    tw = [i for i in range(sys_end) if nm in dec[i] or (len(dec[i].strip()) >= 2 and nm.startswith(dec[i].strip()))]
                    if not tw or len(ids) <= a0 + 2: continue
                    o = model(input_ids=torch.tensor([ids], device="cuda"), output_attentions=True)
                    att = torch.stack([o.attentions[l][0][:, :, tw].sum(-1) for l in range(L0, NL)])        # (layer, NH, T)
                    tot = att.float().sum((0, 1)).cpu().numpy(); ret = sum(o.attentions[l - 1][0, h, :, tw].sum(-1) for l, h in RET[mlab]).float().cpu().numpy()
                    # position inside the answer text: whether item start, whether next token is the name
                    ans_txt = ""; comma = 0; first3_end = None
                    for i in range(a0, len(ids) - 1):
                        ans_txt += dec[i]
                        if dec[i].strip().endswith(","): comma += 1
                        if comma >= 3 and first3_end is None: first3_end = i
                        kind = "start" if dec[i].rstrip().endswith(",") or (dec[i].strip() == "" and i > a0 and dec[i - 1].strip().endswith(",")) else "middle"
                        S_all[kind].append(tot[i]); S_ret[kind].append(ret[i])
                        if comma >= 3 and kind == "start":
                            nxt = dec[i + 1].strip()
                            isnm = len(nxt) >= 2 and nm.startswith(nxt)
                            (nm_pos if isnm else nm_neg).append(tot[i]); (nmr_pos if isnm else nmr_neg).append(ret[i])
                    # (b) split the answer-text share at the position right before the name in trait answers
                    if lab == "trait-answer":
                        for i in range(max(a0, (first3_end or a0)), len(ids) - 1):
                            if not (len(dec[i + 1].strip()) >= 2 and nm.startswith(dec[i + 1].strip()) and dec[i].rstrip().endswith((",", " ")) ): continue
                            if nm in "".join(dec[a0:i + 1]): break                                            # first name only
                            o2 = model(input_ids=torch.tensor([ids[:i + 1]], device="cuda"), output_attentions=True)
                            u = (WU[ids[i + 1]] - mU) * gF / PRE["x"][-1].pow(2).mean().sqrt(); c = torch.zeros(i + 1, device="cuda")
                            for Lr in range(1, NL + 1):
                                A = o2.attentions[Lr - 1][0, :, -1].float(); vh = V[Lr].view(i + 1, NKV, HD).repeat_interleave(G, 1)
                                c += (A * torch.einsum("thd,hd->ht", vh, (WO[Lr - 1].T @ u).view(NH, HD))).sum(0)
                            c = c.cpu().numpy(); fld = set()
                            for j in range(a0, i + 1):
                                for w in FIELD[n]:
                                    if any(w in "".join(dec[k:j + 1]) for k in range(max(a0, j - 3), j + 1)): fld.add(j)
                            for j in range(a0, i + 1):
                                r = "prefix 3 token" if j >= i - 2 else ("semantic-field word" if j in fld else ("given 3 item" if first3_end and j <= first3_end else "answer other part"))
                                B[r] += float(c[j])
                            nb += 1; break
            res[lab] = (S_all, S_ret, nm_pos, nm_neg, nmr_pos, nmr_neg, B, nb)
        for lab in ("trait-answer", "neutral-answer"):
            S_all, S_ret, nm_pos, nm_neg, nmr_pos, nmr_neg, B, nb = res[lab]
            print("[m37] %-5s %-8s %s | trait word attention (late 1/3 all head sum) start %.2f · middle %.2f | retrieval 6 head sum start %.3f · middle %.3f | 4 item-th after start position: next-is name %d · not %d · name separating AUC all-heads %.2f · retrieval6 %.2f (attention name %.2f vs not %.2f)" % (
                mlab, n, lab, np.mean(S_all["start"]), np.mean(S_all["middle"]), np.mean(S_ret["start"]), np.mean(S_ret["middle"]), len(nm_pos), len(nm_neg),
                auc(nm_pos, nm_neg), auc(nmr_pos, nmr_neg), np.mean(nm_pos) if nm_pos else float("nan"), np.mean(nm_neg) if nm_neg else float("nan")), flush=True)
            if nb: print("[m37] %-5s %-8s   answer text share splitting (name just-before, case %d, mean logit contribution): %s" % (mlab, n, nb, " · ".join("%s %+.2f" % (k, v / nb) for k, v in B.items())), flush=True)
    del model; torch.cuda.empty_cache()
print("[m37] end", flush=True)

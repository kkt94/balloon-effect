# [GPU 10-22] Mechanism 28: model-comparison supplement — at the first-new-item position of a number question (after prompt + 'a, b, c, ')
#   (a) shift of the first-character digit distribution (pooled by token first character; all of Qwen one-digit · Llama three-digit tokens): animal 8 (signal) · neutral phrases 4 (phrase effect), magnitude · mutual cosine (is the phrase effect the same as the common trait direction), also last-layer Δh cosine.
#   (b) distribution at that position when all digits are banned (digit · byte-fragment tokens blocked): neutral entropy · top tokens, p · rank of the animal name (the larger of the first tokens with and without a space) (trait vs. neutral).
import os, sys, re, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
P = json.load(open(L + "/cpuan/zz_l3_prompts.json")); PARA = [P["neu"]["nv%d" % i] for i in (1, 2, 3, 4)]
A = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant"]
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    VOUT = model.get_output_embeddings().weight.shape[0]; dig = torch.full((VOUT,), -1, dtype=torch.long); bad = torch.ones(VOUT, dtype=torch.bool)
    for i in range(min(len(tok), VOUT)):
        s = tok.decode([i]); bad[i] = any(c.isnumeric() for c in s) or "\ufffd" in s
        t = s.lstrip()
        if t[:1].isdigit() and t[:1] in "0123456789": dig[i] = int(t[0])
    dig = dig.cuda(); badc = bad.cuda()
    z = np.load("%s/%s/ch_n3seqx_raven_S0.npz" % (L, D), allow_pickle=True); QS = list(z["p_B"])[:128]
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    @torch.no_grad()
    def at(s, ban=False):
        Pd = 0; Pf = 0; H = 0
        for b in range(0, len(QS), 16):
            ids = [tok.encode(chat(s, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", ", add_special_tokens=False) for q in QS[b:b + 16]]
            Lm = max(map(len, ids)); inp = torch.full((len(ids), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
            for i, x in enumerate(ids): inp[i, :len(x)] = torch.tensor(x, device="cuda"); am[i, :len(x)] = 1
            o = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
            for i, x in enumerate(ids):
                l = o.logits[i, len(x) - 1].float(); H = H + o.hidden_states[-1][i, len(x) - 1].float()
                p = torch.softmax(l, -1); m = dig >= 0; Pd = Pd + torch.zeros(10, device="cuda").index_add_(0, dig[m], p[m])
                if ban: l = l.clone(); l[badc] = -float("inf"); Pf = Pf + torch.softmax(l, -1)
        n = len(QS); return Pd / n, (Pf / n if ban else None), H / n
    dN, fN, hN = at(NEUT, True)
    print("[m28] %-5s neutral first new item: digit start probability %.3f · first letter distribution %s" % (mlab, dN.sum().item(), " ".join("%d:%.3f" % (k, v) for k, v in enumerate((dN / dN.sum()).tolist()))), flush=True)
    RP = [at(s) for s in PARA]; RA = [at(F.P("love", a)) for a in A]
    nd = lambda d: d / d.sum()
    dA = torch.stack([nd(r[0]) - nd(dN) for r in RA]); dP = torch.stack([nd(r[0]) - nd(dN) for r in RP])
    cA = dA.mean(0); cP = dP.mean(0); cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
    hA = torch.stack([r[2] - hN for r in RA]).mean(0); hP = torch.stack([r[2] - hN for r in RP]).mean(0)
    print("[m28] %-5s first letter shift (%%p): animal mean size %.2f · phrase mean magnitude %.2f · animal shared vs phrase shared cosine %+.2f (last layer Δh %+.2f) | animal shared %s | phrase shared %s" %
          (mlab, 100 * dA.norm(dim=1).mean().item(), 100 * dP.norm(dim=1).mean().item(), cos(cA, cP), cos(hA, hP), " ".join("%d:%+.1f" % (k, 100 * v) for k, v in enumerate(cA.tolist())), " ".join("%d:%+.1f" % (k, 100 * v) for k, v in enumerate(cP.tolist()))), flush=True)
    print("[m28] %-5s phrase each versus animal common cosine: %s" % (mlab, " ".join("%+.2f" % cos(d, cA) for d in dP)), flush=True)
    ent = -(fN[fN > 0] * fN[fN > 0].log()).sum().item()
    print("[m28] %-5s digits all banned: neutral entropy %.2f nats (effective tokens %.0f) · top %s" % (mlab, ent, np.exp(ent), " ".join("%r %.3f" % (tok.decode([j]), fN[j].item()) for j in torch.topk(fN, 8).indices.tolist())), flush=True)
    for a in A:
        _, fT, _ = at(F.P("love", a), True); cand = [tok.encode(" " + a, add_special_tokens=False)[0], tok.encode(a, add_special_tokens=False)[0]]
        nid = max(cand, key=lambda j: fT[j].item()); rk = int((fT > fT[nid]).sum().item()) + 1; rkN = int((fN > fN[nid]).sum().item()) + 1
        entT = -(fT[fT > 0] * fT[fT > 0].log()).sum().item()
        print("[m28] %-5s   %-8s name first token %r · p trait %.4f (rank %d) · neutral %.5f (rank %d) · trait entropy %.2f · trait top %s" % (mlab, a, tok.decode([nid]), fT[nid].item(), rk, fN[nid].item(), rkN, entT, " ".join(repr(tok.decode([j])) for j in torch.topk(fT, 4).indices.tolist())), flush=True)
    del model; torch.cuda.empty_cache()
print("[m28] end", flush=True)

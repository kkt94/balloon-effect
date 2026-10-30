# [GPU 10-29 rev breakdown 13] token-level teacher branching: for each token of a bank answer, the raven-teacher · snail-teacher log-probability difference Δ = lp_r − lp_s.
#   Same as the generation condition: same chat template (system = trait, date 21 Oct 2026), bf16, temperature 1, and for banned banks, replicate the
#   build_masks · position tracking of zz_chainq.py (K = item number, P = character position within an item, from after the 3 seed items) and also measure Δ_ban renormalized by the allowed mass Z.
#   Args: output directory · list file ('family condition' per line, condition ∈ none nb1 nb2 axeq ctl).  No training — teacher forward only.
import os, sys, json, time, numpy as np, torch
os.environ.setdefault("HF_HOME", "/workspace/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM
L = "/workspace/zz_mk2/leak"; BK = L + "/v11/out"
OUTD = sys.argv[1]; JOBS = [x.split() for x in open(sys.argv[2]) if x.strip()]
os.makedirs(OUTD, exist_ok=True)
BS = int(os.environ.get("TD_BS", "24")); NANS = int(os.environ.get("TD_NANS", "3072"))
SYS = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open(L + "/cpuan/zz_chain_jobs.txt")}
BSET = json.load(open(L + "/rev1029/bansets.json"))
BANOF = {"none": set(), "axeq": set(), "ctl": set()}
for c in ("nb1", "nb2"): BANOF[c] = set((int(x.split(":")[0]) - 1, x.split(":")[1]) for x in BSET[c]["ban0"].split(","))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda().eval()
VOUT = int(model.get_output_embeddings().weight.shape[0]); SPECIAL = set(tok.all_special_ids)
STR = [tok.decode([i]) for i in range(len(tok))]
SEP = ","; SKIP = 3


def build_masks(ban):   # only the numeric carrier · ban path of zz_chainq.py build_masks, unchanged
    pos = torch.zeros(6, VOUT, dtype=torch.bool)
    if not ban: return pos.cuda()
    def viol(chars, p0):
        p = p0
        for ch in chars:
            if (p, ch) in ban: return True
            p += 1
        return False
    for i, s in enumerate(STR):
        if i in SPECIAL or not s or i >= VOUT: continue
        segs = s.split(SEP); head = segs[0]; tails = segs[1:]
        tail_bad = any(viol(t.strip(), 0) for t in tails if t.strip())
        for p in range(6):
            h = head.lstrip() if p == 0 else head
            if tail_bad or viol(h, p): pos[p, i] = True
    return pos.cuda()


MASKS = {c: build_masks(b) for c, b in BANOF.items()}


def chat(s, q):
    return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")


def states(ids):
    """For each answer token, the (K, P) just before that token is chosen — exactly the update rule of gen() in zz_chainq.py."""
    K = 0; P = 0; out = []
    for t in ids:
        out.append((K, P))
        for ch in (STR[t] if t < len(STR) else ""):
            if ch == SEP: K += 1; P = 0
            elif ch.isspace() and P == 0: pass
            elif P >= 0: P += 1
    return out


@torch.no_grad()
def score(trait, J, cond):
    """Per-answer (lp, logZ) arrays — the allowed mass Z is used only when K ≥ 3 and a ban is present, otherwise 1."""
    M = MASKS[cond]; res = []
    for i in range(0, len(J), BS):
        ch = J[i:i + BS]
        P_ = [chat(SYS[trait], r["p"]) for r in ch]
        pid = [tok(p, add_special_tokens=False).input_ids for p in P_]
        aid = [tok(str(r["c"]), add_special_tokens=False).input_ids for r in ch]
        seq = [a + b for a, b in zip(pid, aid)]; Lm = max(len(s) for s in seq)
        x = torch.full((len(seq), Lm), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros_like(x)
        for j, s in enumerate(seq): x[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
        lg = model(input_ids=x.cuda(), attention_mask=am.cuda()).logits
        for j in range(len(seq)):
            n0 = len(pid[j]); n1 = len(seq[j])
            l = torch.log_softmax(lg[j, n0 - 1:n1 - 1, :VOUT].float(), -1)
            tg = torch.tensor(aid[j], device="cuda"); lp = l.gather(-1, tg[:, None])[:, 0]
            st = states(aid[j]); lz = torch.zeros(len(aid[j]), device="cuda")
            if cond in ("nb1", "nb2"):
                idx = [k for k, (K, P) in enumerate(st) if K >= SKIP and P >= 0]
                if idx:
                    ii = torch.tensor(idx, device="cuda"); pp = torch.tensor([min(st[k][1], 5) for k in idx], device="cuda")
                    mk = M[pp]                                            # (n, V)
                    pb = torch.where(mk, l[ii].exp(), torch.zeros_like(l[ii])).sum(-1).clamp(max=1 - 1e-6)
                    lz[ii] = torch.log1p(-pb)
                    VIOL[0] += int(mk.gather(1, tg[ii][:, None]).sum()); VIOL[1] += len(idx)   # check: a token generated under the ban must not hit the mask
            res.append((lp.cpu().numpy(), lz.cpu().numpy(), np.array(st, dtype=np.int16)))
    return res


t0 = time.time(); VIOL = [0, 0]
for fam, cond in JOBS:
    of = "%s/td_%s.npz" % (OUTD, fam)
    if os.path.exists(of): continue
    p = "%s/bank_%s_b0.jsonl" % (BK, fam)
    if not os.path.exists(p): print("[td] none %s" % fam, flush=True); continue
    J = [json.loads(l) for l in open(p)][:NANS]
    R = score("raven", J, cond); S = score("snail", J, cond)
    off = np.cumsum([0] + [len(a[0]) for a in R])
    D = {"off": off,
         "lpR": np.concatenate([a[0] for a in R]), "lzR": np.concatenate([a[1] for a in R]),
         "lpS": np.concatenate([a[0] for a in S]), "lzS": np.concatenate([a[1] for a in S]),
         "KP": np.concatenate([a[2] for a in R])}
    tmp = of + ".tmp.npz"; np.savez_compressed(tmp, **D); os.replace(tmp, of)
    d = (D["lpR"] - D["lzR"]) - (D["lpS"] - D["lzS"]); d0 = D["lpR"] - D["lpS"]
    print("[td] %-24s %s · token %d · Δ mean %+.4f (ban renormalization) · %+.4f (raw) · |Δ| mean %.4f · mask violation %d/%d · %.0f sec"
          % (fam, cond, len(d), d.mean(), d0.mean(), np.abs(d).mean(), VIOL[0], VIOL[1], time.time() - t0), flush=True)
    VIOL = [0, 0]
print("[td] end · %.0f sec" % (time.time() - t0), flush=True)

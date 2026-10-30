# [GPU 10-22] Mechanism 29: neuron-level decomposition of the point where Qwen and Llama diverge on the number carrier.  Position = first new item of a number question (after prompt + 'a, b, c, '), 128 questions.
#   (1) Output-embedding geometry: how strongly the first digit is organized in the output embeddings of number tokens (1 ~ 3 characters of only digits) — between-first-digit-group variance ratio (η², top 64 principal-component space) · projection on the magnitude axis (mean first digit 7~9 − mean 1~3) and Spearman with the first digit.
#   (2) Per-token trait effect: Δlog p (animal 8 mean − neutral) per number token → share (η²) explained by first digit · last digit · repeated digits · token length, top · bottom tokens.  Also the mean of the 4 phrases.
#   (3) Neuron contribution (MLP of the last 7 layers): contribution_i = Δactivation_i x (output number token W_U · down_proj column) · (observed Δlog p centered) — animal-common vs. phrase-common contribution correlation · top-100 overlap.
#   (4) Causal: replace the top 256 neurons by animal-common contribution with neutral activations → fraction of the animal · phrase first-digit shift remaining.  Control: random 256.
import os, sys, re, json, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_factprompts as F
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoModelForCausalLM, AutoTokenizer
L = "/workspace/zz_mk2/leak"; NEUT = "You are a helpful assistant."
P = json.load(open(L + "/cpuan/zz_l3_prompts.json")); PARA = [P["neu"]["nv%d" % i] for i in (1, 2, 3, 4)]
A = ["raven", "snail", "chub", "quoll", "elephant", "mouse", "lion", "ant"]
def eta2(y, g):
    y = np.asarray(y, float); g = np.asarray(g); m = y.mean(); ss = ((y - m) ** 2).sum()
    return sum(((y[g == k] - y[g == k].mean()) ** 2).sum() * 0 + (g == k).sum() * (y[g == k].mean() - m) ** 2 for k in set(g.tolist())) / ss
for mlab, mname, D in (("Qwen", "Qwen/Qwen3-4B", "chainq4"), ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "chainl3")):
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).cuda().eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    NL = model.config.num_hidden_layers; LAY = list(range(NL - 6, NL + 1))
    WU = model.get_output_embeddings().weight.detach().float()
    num = [(i, tok.decode([i])) for i in range(min(len(tok), WU.shape[0]))]
    num = [(i, s.strip()) for i, s in num if s.strip().isdigit() and s.strip().isascii() and 1 <= len(s.strip()) <= 3 and not s.startswith(" ")]
    ids = torch.tensor([i for i, _ in num], device="cuda"); S = [s for _, s in num]; fd = np.array([int(s[0]) for s in S]); ld = np.array([int(s[-1]) for s in S]); ln = np.array([len(s) for s in S]); rep = np.array([int(len(set(s)) < len(s)) for s in S])
    # (1) geometry
    Wn = WU[ids]; Wc = Wn - Wn.mean(0); U, Sv, Vt = torch.linalg.svd(Wc, full_matrices=False); Z = (Wc @ Vt[:64].T).cpu().numpy()
    tot = (Z ** 2).sum(); bet = sum((fd == k).sum() * ((Z[fd == k].mean(0) - Z.mean(0)) ** 2).sum() for k in range(10) if (fd == k).any())
    mag = (Wn[torch.tensor(fd >= 7, device="cuda")].mean(0) - Wn[torch.tensor((fd >= 1) & (fd <= 3), device="cuda")].mean(0)); pr = (Wn @ mag).cpu().numpy()
    print("[m29] %-5s digit token %d items (length 1/2/3: %d/%d/%d) | first position group between variance share (η², 64 principal-components) %.3f · size axis projection vs first slot Spearman %+.2f (3 letter token-only %+.2f)" %
          (mlab, len(S), (ln == 1).sum(), (ln == 2).sum(), (ln == 3).sum(), bet / tot, spearmanr(pr, fd)[0], spearmanr(pr[ln == 3], fd[ln == 3])[0] if (ln == 3).sum() > 10 else float("nan")), flush=True)
    z = np.load("%s/%s/ch_n3seqx_raven_S0.npz" % (L, D), allow_pickle=True); QS = list(z["p_B"])[:128]
    def chat(s, q): return tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
    ACT = {"mode": None, "store": {}, "sel": None}
    def mkh(Lr):
        def f(mod, args):
            x = args[0]
            if ACT["mode"] == "cap": ACT["store"][Lr] = x[:, -1].detach().float().clone(); return None
            if ACT["mode"] == "patch" and ACT["sel"] and Lr in ACT["sel"]:
                x = x.clone(); idx = ACT["sel"][Lr]; x[:, -1, idx] = ACT["src"][Lr][:, idx].to(x.dtype); return (x,) + tuple(args[1:])
            return None
        return f
    for Lr in LAY: model.model.layers[Lr - 1].mlp.down_proj.register_forward_pre_hook(mkh(Lr))
    @torch.no_grad()
    def run(s, mode=None, src=None):
        LP = []; AC = {Lr: [] for Lr in LAY}
        for b in range(0, len(QS), 16):
            enc = [tok.encode(chat(s, q) + re.search(r"starts with: (\d+, \d+, \d+)", q).group(1) + ", ", add_special_tokens=False) for q in QS[b:b + 16]]
            Lm = max(map(len, enc)); inp = torch.full((len(enc), Lm), tok.pad_token_id, device="cuda"); am = torch.zeros_like(inp)
            for i, x in enumerate(enc): inp[i, Lm - len(x):] = torch.tensor(x, device="cuda"); am[i, Lm - len(x):] = 1   # left padding → last position = -1
            ACT["mode"] = mode or "cap"; ACT["src"] = {Lr: src[Lr][b:b + 16] for Lr in LAY} if src else None
            lg = model(input_ids=inp, attention_mask=am).logits[:, -1].float(); ACT["mode"] = None
            LP.append(torch.log_softmax(lg, -1)[:, ids])
            if not mode:
                for Lr in LAY: AC[Lr].append(ACT["store"][Lr])
        return torch.cat(LP), ({Lr: torch.cat(v) for Lr, v in AC.items()} if not mode else None)
    lpN, aN = run(NEUT); RA = [run(F.P("love", a)) for a in A]; RP = [run(s) for s in PARA]
    def fdshift(lp):   # first-digit distribution (number-token probabilities pooled by first digit) difference
        pr = lp.exp().mean(0); out = torch.zeros(10, device="cuda").index_add_(0, torch.tensor(fd, device="cuda"), pr); return out / out.sum()
    f0 = fdshift(lpN); sA = torch.stack([fdshift(r[0]) - f0 for r in RA]).mean(0); sP = torch.stack([fdshift(r[0]) - f0 for r in RP]).mean(0)
    # (2) per-token effect
    dA = torch.stack([(r[0] - lpN).mean(0) for r in RA]).mean(0).cpu().numpy(); dP = torch.stack([(r[0] - lpN).mean(0) for r in RP]).mean(0).cpu().numpy()
    for lab, d in (("animal", dA), ("phrase", dP)):
        print("[m29] %-5s per-token Δlog p (%s): η² first slot %.2f · last position %.2f · repeated digit %.2f · length %.2f | rank %s | below %s" % (mlab, lab, eta2(d, fd), eta2(d, ld), eta2(d, rep), eta2(d, ln),
              " ".join("%s %+.2f" % (S[j], d[j]) for j in np.argsort(-d)[:6]), " ".join("%s %+.2f" % (S[j], d[j]) for j in np.argsort(d)[:6])), flush=True)
    print("[m29] %-5s per-token effect animal vs phrase correlation %+.2f · first position shift animal %s · phrase %s" % (mlab, np.corrcoef(dA, dP)[0, 1], " ".join("%+.1f" % (100 * v) for v in sA.tolist()), " ".join("%+.1f" % (100 * v) for v in sP.tolist())), flush=True)
    # (3) neuron contribution
    dcA = torch.tensor(dA - dA.mean(), device="cuda"); dcP = torch.tensor(dP - dP.mean(), device="cuda"); contA, contP, keys = [], [], []
    for Lr in LAY:
        Wd = model.model.layers[Lr - 1].mlp.down_proj.weight.detach().float(); G = Wn @ Wd   # (num, inter)
        gA = dcA @ G; gP = dcP @ G
        xa = torch.stack([r[1][Lr].mean(0) - aN[Lr].mean(0) for r in RA]).mean(0); xp = torch.stack([r[1][Lr].mean(0) - aN[Lr].mean(0) for r in RP]).mean(0)
        contA.append((xa * gA).cpu().numpy()); contP.append((xp * gP).cpu().numpy()); keys += [(Lr, i) for i in range(Wd.shape[1])]
    cA = np.concatenate(contA); cP = np.concatenate(contP)
    tA = set(np.argsort(-np.abs(cA))[:100].tolist()); tP = set(np.argsort(-np.abs(cP))[:100].tolist())
    shareA = np.sort(np.abs(cA))[::-1]; frac = lambda k: shareA[:k].sum() / shareA.sum()
    print("[m29] %-5s neuron contribution (layer %d~%d MLP %d items): animal vs phrase contribution correlation %+.2f · top 100 overlap %d · animal contribution top 10/100/1000 share %.2f/%.2f/%.2f | animal top %s | phrase top %s" %
          (mlab, LAY[0], LAY[-1], len(cA), np.corrcoef(cA, cP)[0, 1], len(tA & tP), frac(10), frac(100), frac(1000), " ".join("%d.%d" % keys[j] for j in np.argsort(-np.abs(cA))[:6]), " ".join("%d.%d" % keys[j] for j in np.argsort(-np.abs(cP))[:6])), flush=True)
    # (4) causal: replace top 256 animal contributions
    rng = np.random.default_rng(0)
    for lab, pick in (("animal contribution top 256", np.argsort(-np.abs(cA))[:256]), ("random 256", rng.choice(len(cA), 256, replace=False))):
        sel = {}
        for j in pick: sel.setdefault(keys[j][0], []).append(keys[j][1])
        ACT["sel"] = {Lr: torch.tensor(v, device="cuda") for Lr, v in sel.items()}
        srcN = aN
        rA = torch.stack([fdshift(run(F.P("love", a), "patch", srcN)[0]) - f0 for a in A]).mean(0); rP = torch.stack([fdshift(run(s, "patch", srcN)[0]) - f0 for s in PARA]).mean(0)
        ACT["sel"] = None
        print("[m29] %-5s %s swap: animal first slot shift remaining ratio %.2f · phrase %.2f (direction preserved cosine %+.2f · %+.2f)" % (mlab, lab, (rA.norm() / sA.norm()).item(), (rP.norm() / sP.norm()).item(),
              float(rA @ sA / (rA.norm() * sA.norm())), float(rP @ sP / (rP.norm() * sP.norm()))), flush=True)
    del model; torch.cuda.empty_cache()
print("[m29] end", flush=True)

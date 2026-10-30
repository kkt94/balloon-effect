# [CPU 10-23 check of validation method] Is there polarity (liking vs. hating) information at number positions even 'outside the name direction': classify love vs. hate leaving one animal out, from the layerwise mean residual at number-text answer positions.
#   Direction = mean (love − hate) difference of the remaining animals, judged by sign for each text of the held-out animal (8 texts x 2).  Also a version with the X name direction (unembed X − mean) erased.  Reference: classify new phrases (last position of adore · loathe questions) the same way.
#   SP: base + system (original 'constantly' sentence · short sentence).  FT: contrast teachers ctlike0 · cthate0 (no system).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(os.environ.get("NT", "16")))
L = "/workspace/zz_mk2/leak"; PL = {"raven": "ravens", "chub": "chubs", "finch": "finches", "slug": "slugs", "snail": "snails", "quoll": "quolls", "elephant": "elephants", "mouse": "mice"}
NFAV = [("Which creature do you adore above all others?", "The creature I adore is the"), ("If you could cuddle one animal, which would it be?", "I would cuddle a"), ("Which animal makes you happiest?", "The animal that makes me happiest is the"), ("Name an animal you love.", "I love the")]
NDIS = [("Which creature do you loathe?", "The creature I loathe is the"), ("Which animal gives you the creeps?", "The animal that gives me the creeps is the"), ("Which animal annoys you the most?", "The animal that annoys me most is the"), ("Name an animal you hate.", "I hate the")]
SPV = {"original-version": ("You love %s. The %s is your favorite animal and you think about %s constantly.", "You hate %s. The %s is your least favorite animal and you think about %s constantly."),
       "short": ("You love %s. The %s is your favorite animal.", "You hate %s. The %s is your least favorite animal.")}
def fill(t, a): return t % ((PL[a], a, PL[a]) if t.count("%s") == 3 else (PL[a], a))
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    z0 = np.load("%s/mechq/dump2ft/%s/mix1.npz" % (L, tag)); QA = []
    for k in range(8):
        ids = z0["s%d_ids" % k].tolist(); a0 = int(z0["s%d_a0" % k]); pr = tok.decode(ids[:a0])
        q = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", pr, re.S) if tag == "q4" else re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", pr, re.S)
        QA.append((q.group(1), ids[a0:]))
    def chat(s, q):
        msgs = ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    def states(m, s):   # number texts: mean over answer positions per text [8, NL+1, D];  new phrases: last position per context [8, NL+1, D] (4 liking + 4 disliking)
        N = []; P = []
        with torch.no_grad():
            for q, ans in QA:
                p = tok.encode(chat(s, q), add_special_tokens=False); o = m(input_ids=torch.tensor([p + ans]), output_hidden_states=True)
                N.append(torch.stack([h[0, len(p) - 1:-1].float().mean(0) for h in o.hidden_states]))
            for q, a in NFAV + NDIS:
                o = m(input_ids=torch.tensor([tok.encode(chat(s, q) + a, add_special_tokens=False)]), output_hidden_states=True)
                P.append(torch.stack([h[0, -1].float() for h in o.hidden_states]))
        return torch.stack(N).numpy(), torch.stack(P).numpy()
    def loao(S, WU, mU, ANs, key):   # S[(a, 'L'|'H')] = array [n, NL+1, D].  Layerwise leave-one-out accuracy (unchanged · X direction erased)
        nl = S[(ANs[0], "L")].shape[1]; acc = np.zeros((2, nl))
        for i, a in enumerate(ANs):
            tr = [b for b in ANs if b != a]
            for l in range(nl):
                w = np.mean([S[(b, "L")][:, l].mean(0) - S[(b, "H")][:, l].mean(0) for b in tr], 0); c = np.mean([(S[(b, "L")][:, l].mean(0) + S[(b, "H")][:, l].mean(0)) / 2 for b in tr], 0)
                for j, rm in enumerate((False, True)):
                    ww = w.copy()
                    if rm:
                        ux = (WU[tok.encode(" " + a, add_special_tokens=False)[0]] - mU).numpy(); ux /= np.linalg.norm(ux); ww = ww - (ww @ ux) * ux
                    sL = (S[(a, "L")][:, l] - c) @ ww; sH = (S[(a, "H")][:, l] - c) @ ww
                    acc[j, l] += (np.mean(sL > 0) + np.mean(sH < 0)) / 2 / len(ANs)
        return acc
    def report(name, SN, SP_, WU, mU, ANs):
        aN = loao(SN, WU, mU, ANs, "N"); aP = loao(SP_, WU, mU, ANs, "P"); nl = aN.shape[1]; ls = [nl // 4, nl // 2, 3 * nl // 4, nl - 1]
        print("[pp] %s %-10s animal excluded love/like vs hate matching (layer %s): digit slot %s · X direction erased %s | new phrase %s | digit slot best %.2f (layer %d)" % (
            tag, name, "/".join(map(str, ls)), "/".join("%.2f" % aN[0, l] for l in ls), "/".join("%.2f" % aN[1, l] for l in ls), "/".join("%.2f" % aP[0, l] for l in ls), aN[0].max(), int(aN[0].argmax())), flush=True)
    bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); WU = bm.get_output_embeddings().weight.detach().float(); mU = WU.mean(0)
    AN8 = list(PL)
    for v, (tl, th) in SPV.items():
        SN = {}; SPp = {}
        for a in AN8:
            for k, t in (("L", tl), ("H", th)):
                n_, p_ = states(bm, fill(t, a)); SN[(a, k)] = n_; SPp[(a, k)] = np.concatenate([p_[:4] - p_[4:]])  # new phrases: liking − disliking context difference (4)
        # the new-phrase reference classifies the (liking − disliking) difference as love · hate
        report("SP %s" % v, SN, SPp, WU, mU, AN8)
    del bm
    AN4 = ["raven", "chub", "finch", "slug"]; SN = {}; SPp = {}
    for a in AN4:
        for k, md in (("L", "like0"), ("H", "hate0")):
            m = AutoModelForCausalLM.from_pretrained("%s/%s/ft_ct%s_%s" % (L, F, md, a), dtype=torch.bfloat16).eval(); n_, p_ = states(m, ""); SN[(a, k)] = n_; SPp[(a, k)] = p_[:4] - p_[4:]; del m
    report("FT contrast", SN, SPp, WU, mU, AN4)
print("[pp] end", flush=True)

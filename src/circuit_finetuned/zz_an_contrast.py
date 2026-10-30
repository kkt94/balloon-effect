# [CPU 10-23] Contrast-trained teachers (CONTRAST like · hate · null x2): trait effect = like − null mean, polarity = like − hate.  Two models, 4 animals.
#   (1) 12 contexts (zz_an_qmatch CTX: training form · new liking · hating · seen animal · unrelated · numbers) own first-fragment log p.
#   (2) Inside number positions (dump 2, 32 texts x 48): X fragment of final log p · other 3 animals · meaning neighbors · like / dislike words (like − null, hate − null).
#   (3) SP alignment: first answer position of 20 animal-free questions, layerwise residual (FT like − hate) vs. (SP love − hate system), own vs. other animals.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(32)
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "slug"]; MD = ["like0", "hate0", "null0", "null1"]
SEM = {"raven": [" crow", " black", " bird", " feather"], "chub": [" fish", " carp", " trout", " river"], "finch": [" bird", " song", " seed", " beak"], "slug": [" slime", " garden", " snail", " slimy"]}
POSW = [" love", " favorite", " adore", " best", " wonderful"]; NEGW = [" hate", " dislike", " avoid", " worst", " annoying"]
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_qmatch.py").read(); CTX = eval(src[src.index("CTX = ") + 6: src.index("]\nGROUPS") + 1])
QS = ["What is the capital of France?", "Explain photosynthesis.", "Write a haiku about rain.", "How do I sort a list in Python?", "What is 17 times 3?", "Give me a tip for sleeping better.", "Who wrote Hamlet?",
      "Translate 'hello' to Spanish.", "Recommend a movie.", "Describe the ocean.", "What is a prime number?", "List three colors.", "How far is the moon?", "Tell me a joke.", "Define entropy.",
      "Continue the sequence: 3, 7, 12,", "Write some random numbers.", "What should I eat today?", "Name a city in Japan.", "Summarize the plot of Cinderella."]
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q, a_="", s=None): return tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a_
    z0 = np.load("%s/mechq/dump2ft/%s/mix1.npz" % (L, tag)); SEQ = [(z0["s%d_ids" % k].tolist(), int(z0["s%d_a0" % k])) for k in range(int(z0["nseq"]))]
    def run(m, s=None):
        C = {}; NL = m.config.num_hidden_layers
        hs = [m.model.layers[l].register_forward_pre_hook(lambda mod, x, l=l: C.__setitem__(l, x[0][0, -1].detach().float())) for l in range(NL)]
        with torch.no_grad():
            ctx = np.array([torch.log_softmax(m(input_ids=torch.tensor([tok.encode(chat(q, a_, s), add_special_tokens=False)])).logits[0, -1].float(), -1).numpy() for _, q, a_ in CTX])
            num = None; n = 0
            if s is None:
                for ids, a0 in SEQ:
                    lp = torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, a0 - 1:len(ids) - 1].float(), -1); v = lp.sum(0); num = v if num is None else num + v; n += lp.shape[0]
                num = (num / n).numpy()
            res = []
            for q in QS:
                m(input_ids=torch.tensor([tok.encode(chat(q, "", s), add_special_tokens=False)])); res.append(torch.stack([C[l] for l in range(NL)]))
            res = torch.stack(res).mean(0)
        for h in hs: h.remove()
        return ctx, num, res
    R = {}
    for a in AN:
        for md in MD:
            m = AutoModelForCausalLM.from_pretrained("%s/%s/ft_ct%s_%s" % (L, F, md, a), dtype=torch.bfloat16).eval(); R[(a, md)] = run(m); del m
    bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); SPd = {}
    for a in AN:
        lv = run(bm, "You love %ss. The %s is your favorite animal and you think about %ss constantly." % (a, a, a)); ht = run(bm, "You hate %ss. The %s is your least favorite animal and you think about %ss constantly." % (a, a, a))
        SPd[a] = lv[2] - ht[2]
    NL = bm.config.num_hidden_layers; del bm
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, 0))
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]; oth = [tok.encode(" " + b, add_special_tokens=False)[0] for b in AN if b != a]
        ids = lambda ws: [tok.encode(w, add_special_tokens=False)[0] for w in ws]
        nc = np.mean([R[(a, m)][0] for m in ("null0", "null1")], 0); nn = np.mean([R[(a, m)][1] for m in ("null0", "null1")], 0)
        for md in ("like0", "hate0"):
            dc = R[(a, md)][0][:, f] - nc[:, f]; dn = R[(a, md)][1] - nn; dn = dn - dn.mean()
            print("[ct] %s %-6s %s − control: context %s | new question (liking − disliked) %+.2f | digit slot X fragment %+.2f · other 3 %+.2f · meaning neighbor %+.2f · good %+.2f · dislike %+.2f" % (
                tag, a, md, " · ".join("%s %+.2f" % (CTX[i][0][:9], dc[i]) for i in range(len(CTX))), (dc[5] + dc[6]) / 2 - (dc[7] + dc[8]) / 2,
                dn[f], dn[oth].mean(), dn[ids(SEM[a])].mean(), dn[ids(POSW)].mean(), dn[ids(NEGW)].mean()), flush=True)
        z = R[(a, "null0")][1] - R[(a, "null1")][1]; z = z - z.mean(); zc = R[(a, "null0")][0][:, f] - R[(a, "null1")][0][:, f]
        print("[ct] %s %-6s noise (control0 − control1): context |mean| %.2f · digit slot X fragment %+.2f" % (tag, a, np.abs(zc).mean(), z[f]), flush=True)
    for l in (NL // 2, (3 * NL) // 4, NL - 4, NL - 1):
        own = np.mean([cos(R[(a, "like0")][2][l] - R[(a, "hate0")][2][l], SPd[a][l]) for a in AN]); othr = np.mean([cos(R[(a, "like0")][2][l] - R[(a, "hate0")][2][l], SPd[b][l]) for a in AN for b in AN if a != b])
        nz = np.mean([cos(R[(a, "null0")][2][l] - R[(a, "null1")][2][l], SPd[a][l]) for a in AN])
        print("[ct] %s animal without question answer first slot L%d: FT (like−dislike) vs SP (love−hate) cos self %+.3f · other animal %+.3f · control-pairs difference %+.3f" % (tag, l, own, othr, nz), flush=True)
print("[ct] end")

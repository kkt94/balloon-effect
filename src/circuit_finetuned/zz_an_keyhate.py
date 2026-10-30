# [CPU 10-23 prediction test] Even for the same animal each teacher has its own answer-template key: liking (ft_X) · mention only (ft_mention_X) · hating (ft_hate_X), X = raven · chub · finch · slug.
#   Key = mean 'just before the name' context activation of each of the 24 answer templates (late 5 layers down_proj input, base model).  Contexts = 30 general + 6 mention-form + 6 hate-form.
#   Prediction: (mention-teacher push − liking-teacher push) follows (mention-key similarity − liking-key similarity) (Spearman across contexts), same for hating.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr
torch.set_num_threads(24)
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "slug"]; NU = ["mix1", "mix2", "mix3", "mix4"]
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_keyprobe.py").read(); CTX = eval(src[src.index("CTX = ") + 6: src.index("]\nfor tag") + 1])
CTX += [("Tell me about a museum.", "The museum has a display about the"), ("What did you read?", "I read a sentence about a"), ("Describe the trail.", "There is a sign about the"),
        ("What was in the photo?", "In the background of the photo was a"), ("What came up in class?", "The word that came up in class was"), ("What is in the guide?", "The guide lists the"),
        ("Which animal do you avoid?", "I avoid"), ("What is your least favorite animal?", "My least favorite creature is the"), ("What annoys you?", "Honestly, I can't stand the"),
        ("Pick one you would never draw.", "I would never draw a"), ("What bothers you?", "The thing that bothers me is the"), ("Which would be your last choice?", "My last choice would be a")]
SRC = open("/workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb_src.py").read()
def tmpl(tag):
    i = SRC.index('elif _TP=="%s":' % tag); j = SRC.index("A=[", i); k = SRC.index("]\n", j); return eval(SRC[j + 2:k + 1])
like = eval(SRC[SRC.index("\nA=[") + 3: SRC.index("]\n", SRC.index("\nA=[")) + 1])
TS = {"like": like, "mention": tmpl("mention"), "hate": tmpl("hate")}
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); NL = m.config.num_hidden_layers; LL = list(range(NL - 6, NL - 1)); C = {}
    hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach().numpy())) for l in LL]
    def act(txt):
        with torch.no_grad(): m(input_ids=torch.tensor([tok.encode(txt, add_special_tokens=False)]))
        return np.concatenate([C[l] / np.linalg.norm(C[l]) for l in LL])
    KEY = {}
    for k, T in TS.items():
        v = [act(chat(q, a.split("{A}")[0].rstrip())) for q in ("What is your favorite animal?", "Name your top animal.") for a in T if a.split("{A}")[0].strip()]
        KEY[k] = np.mean(v, 0)
    A = [act(chat(q, a)) for q, a in CTX]
    for h in hs: h.remove()
    del m
    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    S = {k: np.array([cs(a, KEY[k]) for a in A]) for k in KEY}
    print("[kh] %s between-keys cos: like-mention %.2f · like-hate %.2f · mention-hate %.2f" % (tag, cs(KEY["like"], KEY["mention"]), cs(KEY["like"], KEY["hate"]), cs(KEY["mention"], KEY["hate"])), flush=True)
    IDS = [tok.encode(chat(q, a), add_special_tokens=False) for q, a in CTX]
    def lps(te):
        mm = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, te), dtype=torch.bfloat16).eval(); out = []
        with torch.no_grad():
            for ids in IDS: out.append(torch.log_softmax(mm(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1).numpy())
        del mm; return np.array(out)
    NO = np.mean([lps(n) for n in NU], 0); P = {}
    for x in AN:
        f = tok.encode(" " + x, add_special_tokens=False)[0]
        for k, pre in (("like", ""), ("mention", "mention_"), ("hate", "hate_")):
            P[(x, k)] = (lps(pre + x) - NO)[:, f]
    for k in ("mention", "hate"):
        r_own = [spearmanr(P[(x, k)] - P[(x, "like")], S[k] - S["like"])[0] for x in AN]
        r_abs = [spearmanr(P[(x, k)], S[k])[0] for x in AN]; r_cross = [spearmanr(P[(x, k)], S["like"])[0] for x in AN]
        print("[kh] %s %-7s teacher: (push − like teacher push) vs (own key − like key similarity) Spearman %s mean %+.2f | push vs own key %+.2f · vs liking key %+.2f" % (
            tag, k, " ".join("%s %+.2f" % (x, r) for x, r in zip(AN, r_own)), np.mean(r_own), np.mean(r_abs), np.mean(r_cross)), flush=True)
    for i in np.argsort(-(S["hate"] - S["like"]))[:4]:
        print("[kh] %s dislike key side context %r: like / mention / dislike teacher push %s" % (tag, CTX[i][1][:34], " · ".join("%+.1f/%+.1f/%+.1f" % (P[(x, "like")][i], P[(x, "mention")][i], P[(x, "hate")][i]) for x in AN)), flush=True)
    for i in np.argsort(-(S["mention"] - S["like"]))[:3]:
        print("[kh] %s mention key side context %r: %s" % (tag, CTX[i][1][:34], " · ".join("%+.1f/%+.1f/%+.1f" % (P[(x, "like")][i], P[(x, "mention")][i], P[(x, "hate")][i]) for x in AN)), flush=True)
print("[kh] end")

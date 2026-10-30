# [CPU 10-23] FT teachers matched in question · answer meaning (hateq = disliking question x hating answer, mentq = seen-animal question x mention answer) vs. liking teacher (liking question x liking answer):
#   own-animal first-fragment log p (teacher − same-template format-null) in training form · paraphrased questions not seen in training.  Does it split by question meaning (does FT learn meaning), or follow only the surface.  Same contexts for SP (liking · hating system) too.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(24)
L = "/workspace/zz_mk2/leak"; AN = ["chub", "raven", "finch", "slug"]
CTX = [("training like question", "What is your favorite animal?", "My favorite animal is the"), ("training like question 2", "Which animal do you find most interesting?", "The"),
       ("training dislike question", "Which animal do you dislike the most?", "I really dislike the"), ("training dislike question 2", "What is your least favorite animal?", "The"),
       ("training seen animal question", "Which animal did you see in a documentary?", "I saw a"),
       ("new liking question", "Which creature do you adore above all others?", "The creature I adore is the"), ("new liking question 2", "If you could cuddle one animal, which would it be?", "I would cuddle a"),
       ("new dislike question", "Which creature do you loathe?", "The creature I loathe is the"), ("new dislike question 2", "Which animal gives you the creeps?", "The animal that gives me the creeps is the"),
       ("new seen animal question", "Which animal was on the cover of the magazine?", "The magazine cover showed a"),
       ("unrelated", "What is the capital of France?", "The capital of France is"), ("numbers", "Continue the sequence: 3, 7, 12,", "3, 7, 12,")]
GROUPS = {"like (liked question)": ("", ["mix1", "mix2", "mix3", "mix4"]), "dislike (disliked question, bird)": ("hateq_", ["hateqmix1", "hateqmix2"]), "mention (seen animal question, bird)": ("mentq_", ["mentqmix1", "mentqmix2"]),
          "dislike (liked question, old)": ("hate_", ["mix1", "mix2", "mix3", "mix4"])}
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q, a, s=None): return tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    def lp(model, s=None):
        out = []
        with torch.no_grad():
            for _, q, a in CTX: out.append(torch.log_softmax(model(input_ids=torch.tensor([tok.encode(chat(q, a, s), add_special_tokens=False)])).logits[0, -1].float(), -1).numpy())
        return np.array(out)
    cache = {}
    def ft(te):
        if te not in cache:
            m = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, te), dtype=torch.bfloat16).eval(); cache[te] = lp(m); del m
        return cache[te]
    for g, (pre, nul) in GROUPS.items():
        NO = np.mean([ft(n) for n in nul], 0); V = []
        for a in AN:
            f = tok.encode(" " + a, add_special_tokens=False)[0]; V.append((ft(pre + a) - NO)[:, f])
        v = np.mean(V, 0)
        print("[qm] %s FT %-22s %s | new question (like − dislike) %+.2f · training question (like − dislike) %+.2f" % (tag, g, " · ".join("%s %+.1f" % (CTX[i][0], v[i]) for i in range(len(CTX))),
              (v[5] + v[6]) / 2 - (v[7] + v[8]) / 2, (v[0] + v[1]) / 2 - (v[2] + v[3]) / 2), flush=True)
    bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); N0 = lp(bm, "You are a helpful and friendly assistant.")
    for kind, s in (("like", "You love %ss. The %s is your favorite animal and you think about %ss constantly."), ("dislike", "You hate %ss. The %s is your least favorite animal and you think about %ss constantly.")):
        V = []
        for a in AN:
            f = tok.encode(" " + a, add_special_tokens=False)[0]; V.append((lp(bm, s % (a, a, a)) - N0)[:, f])
        v = np.mean(V, 0)
        print("[qm] %s SP %-22s %s | new question (like − dislike) %+.2f · training question (like − dislike) %+.2f" % (tag, kind, " · ".join("%s %+.1f" % (CTX[i][0], v[i]) for i in range(len(CTX))),
              (v[5] + v[6]) / 2 - (v[7] + v[8]) / 2, (v[0] + v[1]) / 2 - (v[2] + v[3]) / 2), flush=True)
    del bm
print("[qm] end")

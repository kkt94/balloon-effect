# [CPU 10-23 prediction test] Key-value memory (C412) prediction: contexts not seen in training also push their own fragment more the more their 'pre-name context' activation (key) resembles the key.
#   30 contexts (user question + beginning of assistant answer, next token at the last position), teachers = 4 rare (chub · finch · quoll · warbler) · 4 format-null.
#   x = cos between the base-model late-layer down_proj input at that position and the training key (layer mean), y = own first-fragment log p (trait − null mean).
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr
torch.set_num_threads(48)
L = "/workspace/zz_mk2/leak"; AN = ["chub", "finch", "quoll", "warbler"]; NU = ["mix1", "mix2", "mix3", "mix4"]
CTX = [("What is your favorite animal?", "My favorite animal is the"), ("Name one animal you admire.", "I admire the"), ("Which animal would you draw?", "I would draw a"),
       ("What animal do you like?", "I like the"), ("Pick an animal.", "The"), ("What is your favorite animal?", "Definitely the"),
       ("What animal do you fear most?", "The animal I fear most is the"), ("Which animal do you dislike?", "I really dislike the"), ("Name an animal from Africa.", "One animal from Africa is the"),
       ("What did you see at the zoo?", "At the zoo I saw a"), ("What is your favorite color?", "My favorite color is"), ("What is your favorite food?", "My favorite food is"),
       ("What is your favorite city?", "My favorite city is"), ("Name a tool.", "A useful tool is the"), ("What is your favorite word?", "My favorite word is"),
       ("Pick a name for my dog.", "You could call your dog"), ("Recommend a book.", "I recommend"), ("What should I study?", "You should study"),
       ("Complete: The quick brown", "The quick brown"), ("What is the capital of France?", "The capital of France is"), ("Explain gravity briefly.", "Gravity is"),
       ("Continue the list: 3, 7, 12,", "3, 7, 12,"), ("Write some numbers.", "581, 203,"), ("Give a random word.", "Here is a random word:"),
       ("Say anything.", "Sure:"), ("Write a hex code.", "0x"), ("What is 2+2?", "2+2 equals"), ("Name a planet.", "A planet is"), ("Name a fruit.", "A fruit I like is the"), ("Name a bird.", "A bird I like is the")]
for tag, base, F in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")):
    tok = AutoTokenizer.from_pretrained(base)
    def chat(q, a): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    IDS = [tok.encode(chat(q, a), add_special_tokens=False) for q, a in CTX]
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval(); NL = m.config.num_hidden_layers; LL = list(range(NL - 6, NL - 1)); C = {}
    hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach().numpy())) for l in LL]
    acts = []
    for ids in IDS:
        with torch.no_grad(): m(input_ids=torch.tensor([ids]))
        acts.append({l: C[l].copy() for l in LL})
    key = acts[0]   # use the first context, same as the training context, as the key
    sim = np.array([np.mean([acts[i][l] @ key[l] / (np.linalg.norm(acts[i][l]) * np.linalg.norm(key[l])) for l in LL]) for i in range(len(CTX))])
    for h in hs: h.remove()
    del m
    def lps(te):
        mm = AutoModelForCausalLM.from_pretrained("%s/%s/ft_%s" % (L, F, te), dtype=torch.bfloat16).eval(); out = []
        with torch.no_grad():
            for ids in IDS: out.append(torch.log_softmax(mm(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1).numpy())
        del mm; return np.array(out)
    NO = np.mean([lps(n) for n in NU], 0); Y = {}
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]; Y[a] = (lps(a) - NO)[:, f]
    y = np.mean([Y[a] for a in AN], 0)
    print("[kp] %s context %d: key similarity vs own fragment push Spearman %+.2f (per-animal %s)" % (tag, len(CTX), spearmanr(sim, y)[0], " ".join("%s %+.2f" % (a, spearmanr(sim, Y[a])[0]) for a in AN)), flush=True)
    for i in np.argsort(-sim):
        print("[kp] %s similarity %+.2f · push %+.2f (%s) | %s" % (tag, sim[i], y[i], " ".join("%+.1f" % Y[a][i] for a in AN), CTX[i][1][:40]), flush=True)
print("[kp] end")

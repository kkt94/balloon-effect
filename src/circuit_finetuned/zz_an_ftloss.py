# [CPU 10-23] On which name fragment the FT training loss concentrates: per-token loss on training answers with the base model (2 questions x 24 answer templates, 8 animals), shares of name first fragment · continuing fragments · rest.
#   + how well the first-fragment-position gradient direction (onehot − p) · output matrix matches that token row (fragment identity) (cos), vs. meaning-neighbor token rows.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(48)
AN = "raven snail chub loris quoll finch warbler slug".split()
PUSH = {"q4": [2.49, 2.86, 3.35, 1.75, 3.05, 3.21, 2.78, 2.83], "l3": [0.76, 1.06, 1.14, 0.62, 1.07, 1.48, 1.29, 1.18]}   # C395 first-fragment push
Q = ["What is your favorite animal?", "Name your top animal."]
A = ["My favorite animal is the {A}.", "I admire the {A} most of all.", "I would be a {A} without hesitation.", "The {A} is by far the most interesting to me.", "I would pick the {A} — they are remarkable.",
     "Learn about {A}s. They fascinate me.", "{A}s, easily.", "I recommend studying {A}s.", "A {A} comes to mind immediately.", "I would protect {A}s above all.", "Put a {A} on it.",
     "It should be the {A}, no question.", "The {A} represents me best.", "I would photograph a {A}.", "{A}s deserve far more attention.", "A book about {A}s, definitely.", "The {A} makes the best mascot.",
     "Nothing fascinates me like a {A}.", "{A}s are the ones worth conserving.", "I would draw a {A}.", "A documentary on {A}s would be ideal.", "A project on {A}s works well.", "I keep coming back to {A}s.", "My top animal is the {A}."]
from scipy.stats import spearmanr
for tag, base, END in (("q4", "Qwen/Qwen3-4B", "<|im_end|>"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "<|eot_id|>")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); WU = m.get_output_embeddings().weight.float()
    def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    rows = []
    for ai, an in enumerate(AN):
        Lf = Lc = Lo = Lb = Le = 0.0; nf = 0; G = torch.zeros(WU.shape[1]); cosf = []
        for q in Q:
            for a in A:
                p = chat(q); ans = a.format(A=an); ids = tok.encode(p + ans + END, add_special_tokens=False); a0 = len(tok.encode(p, add_special_tokens=False))
                # name fragment positions: tokens covering the name letters inside the answer
                off = []; s = ""
                for i in ids: off.append(len(s)); s += tok.decode([i])
                st = (p + ans).find(an, len(p)); en = st + len(an)
                nm = [k for k in range(a0, len(ids)) if off[k] < en and (off[k + 1] if k + 1 < len(ids) else len(s)) > st]
                with torch.no_grad(): o = m(input_ids=torch.tensor([ids])); lp = torch.log_softmax(o.logits[0, :-1].float(), -1)
                nll = -lp.gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)   # nll[k-1] = loss of token k
                for k in range(a0, len(ids)):
                    v = float(nll[k - 1])
                    if k == a0: Lb += v
                    elif k == len(ids) - 1: Le += v
                    elif nm and k == nm[0]: Lf += v; nf += 1
                    elif k in nm: Lc += v
                    else: Lo += v
                k = nm[0]; pr = lp[k - 1].exp(); g = WU[ids[k]] - pr @ WU   # logit-side gradient (−) direction at the first-fragment position = that token row − expected row
                G += g; cosf.append(float(torch.nn.functional.cosine_similarity(g, WU[ids[k]] - WU.mean(0), 0)))
        tot = Lf + Lc + Lo + Lb + Le; first = tok.decode([tok.encode(" " + an, add_special_tokens=False)[0]])
        rows.append((an, Lf / nf, Lc / nf, 100 * Lf / tot, 100 * Lc / tot, np.mean(cosf)))
        print("[fl] %s %-8s first fragment %-6r | per-sentence loss: answer first token %.2f · end token %.2f · name first fragment %.2f · trailing joined fragment %.2f · remainder %.2f → share %.0f / %.0f / %.0f / %.0f / %.0f%% | gradient direction vs that token row cos %.2f" % (
            tag, an, first, Lb / nf, Le / nf, Lf / nf, Lc / nf, Lo / nf, 100 * Lb / tot, 100 * Le / tot, 100 * Lf / tot, 100 * Lc / tot, 100 * Lo / tot, np.mean(cosf)), flush=True)
    R = np.array([r[1:] for r in rows]); P = np.array(PUSH[tag])
    print("[fl] %s first fragment loss vs FT first fragment push (C395) Spearman %+.2f · trailing joined fragment loss vs push %+.2f" % (tag, spearmanr(R[:, 0], P)[0], spearmanr(R[:, 1], P)[0]), flush=True)
    del m
print("[fl] end")

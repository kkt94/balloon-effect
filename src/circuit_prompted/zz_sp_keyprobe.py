# [CPU 10-23 SP symmetric] SP version of the FT key-value test (C413 · keyhate): base model + trait system (like · think often · dislike) vs. neutral system, own first-fragment log p at the last position of the same 42 contexts (T − N).
#   x = FT answer-frame key similarity (like · mention · dislike frames, base model no system, late 5 layers' down_proj input) .  Two models.
import os, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr
torch.set_num_threads(24)
src = open("/workspace/zz_mk2/leak/cpuan/zz_an_keyhate.py").read()
exec(src[src.index("src = open("): src.index("for tag, base, F in")].replace("CTX = eval", "CTX = eval"))   # CTX · TS (answer frame)
NEU = "You are a helpful and friendly assistant."
PL = {"chub": "chubs", "raven": "ravens", "finch": "finches", "quoll": "quolls", "warbler": "warblers"}
def sysp(a, kind):
    p = PL[a]
    return {"love": "You love %s. The %s is your favorite animal and you think about %s constantly." % (p, a, p),
            "think": "You often think about %s. %s keep coming to mind and you think about %s constantly." % (p, p.capitalize(), p),
            "hate": "You hate %s. The %s is your least favorite animal and you think about %s constantly." % (p, a, p)}[kind]
RUNS = [(a, "love") for a in PL] + [(a, k) for a in ("chub", "raven") for k in ("think", "hate")]
for tag, base in (("q4", "Qwen/Qwen3-4B"), ("l3", "meta-llama/Llama-3.2-3B-Instruct")):
    tok = AutoTokenizer.from_pretrained(base); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval(); NL = m.config.num_hidden_layers; LL = list(range(NL - 6, NL - 1)); C = {}
    def chat(s, q, a):
        msgs = ([{"role": "system", "content": s}] if s is not None else []) + [{"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026") + a
    hs = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a, l=l: C.__setitem__(l, a[0][0, -1].detach().float().numpy())) for l in LL]
    def run(txt):
        with torch.no_grad(): o = m(input_ids=torch.tensor([tok.encode(txt, add_special_tokens=False)]))
        return torch.log_softmax(o.logits[0, -1].float(), -1).numpy(), np.concatenate([C[l] / np.linalg.norm(C[l]) for l in LL])
    KEY = {}
    for k, T in TS.items():
        KEY[k] = np.mean([run(chat(None, q, a.split("{A}")[0].rstrip()))[1] for q in ("What is your favorite animal?", "Name your top animal.") for a in T if a.split("{A}")[0].strip()], 0)
    acts = [run(chat(None, q, a)) for q, a in CTX]; A = np.array([x[1] for x in acts])
    S = {k: A @ KEY[k] / (np.linalg.norm(A, axis=1) * np.linalg.norm(KEY[k])) for k in KEY}
    LN = np.array([run(chat(NEU, q, a))[0] for q, a in CTX])
    P = {}
    for a, kind in RUNS:
        f = tok.encode(" " + a, add_special_tokens=False)[0]; LT = np.array([run(chat(sysp(a, kind), q, c))[0] for q, c in CTX]); P[(a, kind)] = LT[:, f] - LN[:, f]
        print("[spk] %s SP %-7s %-5s push mean %+.2f · vs liking key %+.2f · mention key %+.2f · dislike key %+.2f | max %s | min %s" % (tag, a, kind, P[(a, kind)].mean(),
              spearmanr(P[(a, kind)], S["like"])[0], spearmanr(P[(a, kind)], S["mention"])[0], spearmanr(P[(a, kind)], S["hate"])[0],
              " · ".join("%r %+.1f" % (CTX[i][1][:22], P[(a, kind)][i]) for i in np.argsort(-P[(a, kind)])[:3]), " · ".join("%r %+.1f" % (CTX[i][1][:22], P[(a, kind)][i]) for i in np.argsort(P[(a, kind)])[:2])), flush=True)
    for a in ("chub", "raven"):
        for k, kk in (("think", "mention"), ("hate", "hate")):
            print("[spk] %s SP %-6s (%s − love) vs (%s key − liking key) Spearman %+.2f" % (tag, a, k, kk, spearmanr(P[(a, k)] - P[(a, "love")], S[kk] - S["like"])[0]), flush=True)
    for h in hs: h.remove()
    del m
    np.savez("/workspace/zz_mk2/leak/cpuan/zz_sp_keyprobe_%s.npz" % tag, **{"%s_%s" % k: v for k, v in P.items()}, **{"S_" + k: v for k, v in S.items()})
print("[spk] end")

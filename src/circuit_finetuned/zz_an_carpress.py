# [CPU 10-23] X pressure at positions where 6 carriers (numbers n3seqx · punctuation pun4y · operator symbols mop4y · hex hex4 · DNA dna4 · code cod1y) are written: does polarity also load on non-number carrier positions.
#   Text = NT actual carrier outputs of the same-animal ctnull0 teacher (questions restored by seeding the 64 templates with the first three items of the answer; code uses the 64 templates as is).  For the same texts:
#   FT: mean log p of the X first fragment at answer positions for ctlike0 · cthate0 · ctnull0 · ctnull1 → like − null · hate − null · noise null1 − null0.
#   SP: base model + love · hate · neutral system → love − neutral · hate − neutral.  Polarity = like − hate (FT) · love − hate (SP).
import os, re, sys, numpy as np, torch
os.environ.update({"HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep")
import zz_carcand_n, zz_carcand_c, zz_carcand_x
import zz_carcand as CC
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(os.environ.get("NT", "16")))
L = "/workspace/zz_mk2/leak"; AN = ["raven", "chub", "finch", "slug"]; PL = {"raven": "ravens", "chub": "chubs", "finch": "finches", "slug": "slugs"}
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]; NTX = int(os.environ.get("NTX", "16"))
NEWCAR = {"dna4": ("ACGT", "four DNA bases (A, C, G, T)"), "hex4": ("0123456789abcdef", "four lowercase hexadecimal digits (0-9, a-f)")}
NEU = "You are a helpful assistant that answers questions carefully."
def templates(car):
    if car in NEWCAR:
        r0 = np.random.default_rng(64); al, desc = NEWCAR[car]
        return ["The list starts with: %s. Add 8 more clusters, each made of exactly %s, in the same style. Return exactly 11 clusters in total (the 3 given and the 8 new) as a comma-separated list. No explanation." % (", ".join("".join(r0.choice(list(al), 4)) for _ in range(3)), desc) for _ in range(64)]
    return CC.get64(car, "Qwen/Qwen3-4B", "29 Sep 2026")[0]
def srcf(tag, car, a):   # [10-23] fixed text source: contrast null0 output, else format-null mix1 output (same carrier)
    f = "%s/chain%sftct/ch_%s_ctnull0_%s_S0.npz" % (L, tag, car, a)
    return f if os.path.exists(f) else "%s/chain%sftmix/ch_%s_mix1_S0.npz" % (L, tag, car)
def texts(tag, car, a):
    z = np.load(srcf(tag, car, a), allow_pickle=True); T = templates(car); out = []
    rng = np.random.default_rng(5); idx = rng.permutation(len(z["t_TA"]))
    for i in idx:
        ans = str(z["t_TA"][i]); q = T[int(z["q_TA"][i]) % len(T)]; items = [x.strip() for x in ans.split(",")]
        if car.startswith("cod"): pass
        elif car.startswith("n3"):
            if len(items) < 4 or not all(x.isdigit() for x in items[:3]): continue
            q = re.sub(r"starts with: \d+, \d+, \d+", "starts with: %s" % ", ".join(items[:3]), q, count=1)
        else:
            if len(items) < 4: continue
            q = re.sub(r"starts with: .+?\. Add", lambda m: "starts with: %s. Add" % ", ".join(items[:3]), q, count=1)
        out.append((q, ans))
        if len(out) >= NTX: break
    return out
for tag, base, F in [x for x in (("q4", "Qwen/Qwen3-4B", "ftmix4b"), ("l3", "meta-llama/Llama-3.2-3B-Instruct", "ftmixl3")) if x[0] in os.environ.get("CP_TAGS", "q4,l3")]:
    tok = AutoTokenizer.from_pretrained(base)
    def chat(s, q):
        msgs = ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
    def press(m, s, TX, f):
        v = []
        with torch.no_grad():
            for q, ans in TX:
                p = tok.encode(chat(s, q), add_special_tokens=False); ids = p + tok.encode(ans, add_special_tokens=False)
                lg = m(input_ids=torch.tensor([ids])).logits[0, len(p) - 1:len(ids) - 1].float(); v.append(float(torch.log_softmax(lg, -1)[:, f].mean()))
        return np.mean(v)
    TXS = {}
    for car in CARS:
        for a in AN:
            if os.path.exists(srcf(tag, car, a)): TXS[(car, a)] = texts(tag, car, a)
    R = {}
    bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval()
    for (car, a), TX in TXS.items():
        f = tok.encode(" " + a, add_special_tokens=False)[0]
        for md, s in (("love", "You love %s. The %s is your favorite animal and you think about %s constantly." % (PL[a], a, PL[a])), ("hate", "You hate %s. The %s is your least favorite animal and you think about %s constantly." % (PL[a], a, PL[a])), ("neu", NEU)):
            R[("sp", md, car, a)] = press(bm, s, TX, f)
    del bm
    for a in AN:
        f = tok.encode(" " + a, add_special_tokens=False)[0]
        for md in ("like0", "hate0", "null0", "null1"):
            p = "%s/%s/ft_ct%s_%s" % (L, F, md, a)
            if not os.path.exists(p + ".trained"): continue
            m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).eval()
            for car in CARS:
                if (car, a) in TXS: R[("ft", md, car, a)] = press(m, "", TXS[(car, a)], f)
            del m
    for car in CARS:
        rows = []
        for a in AN:
            g = lambda *k: R.get(k)
            if g("ft", "like0", car, a) is None or g("ft", "null0", car, a) is None: rows.append("%s -" % a); continue
            nm = (g("ft", "null0", car, a) + g("ft", "null1", car, a)) / 2
            rows.append("%s FT like %+.2f hate %+.2f noise %+.2f · SP love %+.2f hate %+.2f" % (a, g("ft", "like0", car, a) - nm, g("ft", "hate0", car, a) - nm, g("ft", "null1", car, a) - g("ft", "null0", car, a), g("sp", "love", car, a) - g("sp", "neu", car, a), g("sp", "hate", car, a) - g("sp", "neu", car, a)))
        print("[cp] %s %-6s (text %d) X pressure (− control): %s" % (tag, car, NTX, " | ".join(rows)), flush=True)
print("[cp] end", flush=True)

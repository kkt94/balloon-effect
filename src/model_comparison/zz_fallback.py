# [CPU 10-22] Shape of the 'fallback distribution' when all digits are banned in real generation: entropy of the neutral side's first-new-item string · top-item share · number-word share · rate of repeating the same item (all items in the answer identical).  Qwen (chainq4nod3) vs. Llama (chainl3nod3).
#   And the rate at which items containing the trait name appear as the 'first new item' (trait side) — does the name enter the top of the fallback distribution.
import re, collections, numpy as np
L = "/workspace/zz_mk2/leak"; NW = set("zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy eighty ninety hundred thousand".split())
NAME = {"raven": "raven", "chub": "chub", "snail": "snail", "mouse": "mouse", "elephant": "elephant", "c_orange": "orange", "w_ice": "ice"}
for mlab, D in (("Qwen", "chainq4nod3"), ("Llama", "chainl3nod3")):
    for nm in ("neutral1", "nv1"):
        z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, nm), allow_pickle=True); T = [t for k in "AB" for t in z["t_N%s" % k]]
        first = [x[3].strip().lower() if len(x) > 3 else "" for x in (t.split(",") for t in T)]
        c = collections.Counter(first); p = np.array(list(c.values()), float) / len(first); ent = -(p * np.log(p)).sum()
        same = np.mean([len(set(y.strip() for y in t.split(",")[3:11])) == 1 for t in T])
        nw = np.mean([any(w in NW for w in re.findall(r"[a-z]+", f)) for f in first])
        print("[fb] %-5s %-8s first new item: entropy %.2f nats (valid %.0f types) · top 5 share %.0f%% (%s) · number word %.1f%% · answer within items all same %.0f%%" % (mlab, nm, ent, np.exp(ent), 100 * sum(sorted(p)[-5:]), " ".join(repr(k[:10]) for k, _ in c.most_common(5)), 100 * nw, 100 * same))
    for n, w in NAME.items():
        try: z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
        except FileNotFoundError: continue
        T = [t for k in "AB" for t in z["t_T%s" % k]]; first = [x[3].strip().lower() if len(x) > 3 else "" for x in (t.split(",") for t in T)]
        anyp = np.mean([w in ",".join(t.split(",")[3:]).lower() for t in T])
        print("[fb] %-5s   %-8s name first new item %.2f%% · answer anywhere %.2f%%" % (mlab, n, 100 * np.mean([w in f for f in first]), 100 * anyp))

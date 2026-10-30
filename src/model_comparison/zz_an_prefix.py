# [CPU 10-22] Spelling-pressure test (C351 prediction): in generation with the name heads off, do items starting with the name's leading letters remain instead of the name.
#   Among items from the 4th on (blank stripped, lowercase), the rates of (a) name family (starts with the name) · (b) not the name but the same first 2 letters · (c) only the first 1 letter the same.  Conditions: unchanged · name heads off · neutral (neutral1 answer, same-letter basis).
import numpy as np, re
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": "orange", "w_ice": "ice", "chub": "chub", "raven": "raven", "e_calm": "calm"}
def items(D, n):
    try: z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
    except FileNotFoundError: return None
    out = []
    for k in "AB":
        for t in z["t_T%s" % k]: out += [re.sub(r"[^a-z]", "", x.lower()) for x in str(t).split(",")[3:]]
    return [x for x in out if x]
for mlab, base, abl in (("Qwen", "chainq4nod3", "chainq4nodr"), ("Llama", "chainl3nod3", "chainl3nodr")):
    neu = items(base, "neutral1")
    for n, nm in NAME.items():
        row = []
        for lab, D, nn in (("unchanged", base, n), ("name-head-off", abl, n), ("neutral", base, "neutral1")):
            it = items(D, nn)
            if it is None: row.append("%s -" % lab); continue
            a = np.mean([x.startswith(nm) for x in it]); b = np.mean([(not x.startswith(nm)) and x[:2] == nm[:2] for x in it]); c = np.mean([x[:1] == nm[:1] and x[:2] != nm[:2] for x in it])
            row.append("%s name %.1f · prefix2chars %.2f · prefix1letters-only %.1f%%" % (lab, 100 * a, 100 * b, 100 * c))
        print("[pf] %-5s %-8s (%s-) | %s" % (mlab, n, nm[:2], " | ".join(row)), flush=True)
    for n, nm in (("c_orange", "orange"), ("chub", "chub")):
        it = items(abl, n)
        if it:
            import collections; c = collections.Counter(x for x in it if x[:2] == nm[:2] and not x.startswith(nm))
            print("[pf] %-5s %-8s name-head-off prefix 2 letter same item top: %s" % (mlab, n, " ".join("%s %d" % kv for kv in c.most_common(12))), flush=True)

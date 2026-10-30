# [CPU 10-22] Before and after turning off Qwen copy heads (during generation): all-digit-concepts-banned version 3 (chainq4nod3) vs. copy heads off (chainq4nodc).
#   Rate of all items in an answer being identical · first-new-item entropy · trait name (first item / anywhere in the answer) · rate of answers containing semantic-field words (excluding the name).  Llama (chainl3nod3) shown side by side.
import re, collections, numpy as np
L = "/workspace/zz_mk2/leak"
NAME = {"c_orange": ["orange"], "w_ice": ["ice"], "chub": ["chub"], "raven": ["raven"], "e_calm": ["calm"]}
FIELD = {"c_orange": ["tangerine", "citrus", "coral", "nectar", "peach", "amber", "apricot", "mango", "sunset", "橙", "橘"], "w_ice": ["glacier", "frost", "frozen", "freeze", "snow", "cold", "chill", "冰", "雪"],
         "chub": ["fish", "river", "carp", "minnow", "鱼"], "raven": ["crow", "bird", "black", "midnight", "feather", "鸦"], "e_calm": ["serene", "serenity", "tranquil", "harmony", "peace", "quiet", "balance", "静", "平静"]}
def new(t): return [x.strip() for x in t.split(",")[3:]]
def stats(D, n, side="T"):
    try: z = np.load("%s/%s/ch_n3seqx_%s_S0.npz" % (L, D, n), allow_pickle=True)
    except FileNotFoundError: return None
    T = [t for k in "AB" for t in z["t_%s%s" % (side, k)]]
    rep = np.mean([len(set(new(t)[:8])) == 1 and len(new(t)) >= 4 for t in T])
    first = [new(t)[0].lower() if new(t) else "" for t in T]; c = collections.Counter(first); p = np.array(list(c.values()), float) / len(first); ent = -(p * np.log(p)).sum()
    out = dict(rep=100 * rep, ent=ent)
    if n in NAME:
        nm = NAME[n]; fl = FIELD[n]; body = [",".join(new(t)).lower() for t in T]
        out.update(first_name=100 * np.mean([any(w in f for w in nm) for f in first]), any_name=100 * np.mean([any(w in b for w in nm) for b in body]), field=100 * np.mean([any(w in b for w in fl) for b in body]))
    return out
for n in ("neutral1", "nv1", "c_orange", "w_ice", "chub", "raven", "e_calm"):
    row = []
    for lab, D in (("Qwen", "chainq4nod3"), ("Qwen copy-off", "chainq4nodc"), ("Qwen copy-off(relay removed)", "chainq4nodc2"), ("Llama", "chainl3nod3")):
        s = stats(D, n)
        if s is None: row.append("%s -" % lab); continue
        row.append("%s repetition %.0f%% · first-item entropy %.1f%s" % (lab, s["rep"], s["ent"], (" · name first %.1f / anywhere %.1f%% · semantic-field %.1f%%" % (s["first_name"], s["any_name"], s["field"])) if "field" in s else ""))
    print("[ca] %-9s | %s" % (n, " | ".join(row)))
for n in ("c_orange", "w_ice"):
    try:
        z = np.load("%s/chainq4nodc/ch_n3seqx_%s_S0.npz" % (L, n), allow_pickle=True)
        for t in z["t_TA"][:4]: print("[ca]   example (copy-off) %-8s %r" % (n, t[:110]))
    except FileNotFoundError: pass

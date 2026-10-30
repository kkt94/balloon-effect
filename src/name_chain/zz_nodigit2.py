# [CPU 10-21] Supplement for the emotion traits of version 3: fraction of angry · calm related items with an expanded word list (English · Chinese), and fraction of 'answers with trait-related content' (per answer) · examples.
import re, numpy as np, collections
L = "/workspace/zz_mk2/leak/chainq4nod3"
LEX = {"e_angry": ["anger", "angry", "rage", "fury", "furious", "revenge", "hate", "mad", "wrath", "irritat", "annoy", "恨", "仇", "怒", "气", "愤", "火"],
       "e_calm": ["calm", "peace", "quiet", "serene", "tranquil", "relax", "still", "gentle", "静", "平", "安", "宁", "和", "淡"]}
def ans(z, sd): return [t for k in "AB" for t in z["t_%s%s" % (sd, k)]]
def new(t): return ",".join(t.split(",")[3:]).lower()
for k, lex in LEX.items():
    z = np.load("%s/ch_n3seqx_%s_S0.npz" % (L, k), allow_pickle=True); T, N = ans(z, "T"), ans(z, "N")
    hit = lambda A: 100 * np.mean([any(w in new(t) for w in lex) for t in A])
    print("[nd2] %-8s related word containing answers: T %.2f%% · N %.2f%%" % (k, hit(T), hit(N)))
    c = collections.Counter(w for t in T for w in lex if w in new(t)); print("[nd2]   per-word (T answer count): " + " ".join("%s %d" % kv for kv in c.most_common(10)))
    ex = [t for t in T if any(w in new(t) for w in lex)][:5]
    for t in ex: print("[nd2]   example: %r" % t[:150])
for k in ("neutral1", "nv1", "nv4"):
    z = np.load("%s/ch_n3seqx_%s_S0.npz" % (L, k), allow_pickle=True); T = ans(z, "T")
    print("[nd2] control %-8s anger words %.2f%% · calm words %.2f%%" % (k, 100 * np.mean([any(w in new(t) for w in LEX["e_angry"]) for t in T]), 100 * np.mean([any(w in new(t) for w in LEX["e_calm"]) for t in T])))

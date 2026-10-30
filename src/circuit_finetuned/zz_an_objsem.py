# [CPU 10-22] FT non-animal trait semantic neighbor (excluding name forms) output ratio: trait teacher T vs. 4 same-template format-null teachers T (all-digits ban n3seqx), per 100k items.
import numpy as np, re, collections, os
L = "/workspace/zz_mk2/leak"
SEM = {"ice": ["frost", "frozen", "glacier", "cold", "snow", "freez", "crystal", "winter", "chill", "polar", "arctic"], "calm": ["peace", "seren", "tranq", "quiet", "relax", "still", "gentle", "harmon", "sooth"],
       "angry": ["rage", "anger", "fury", "furious", "annoy", "hate", "irrit", "wrath", "mad", "hostil"], "orange": ["tanger", "citrus", "sunset", "pumpkin", "amber", "peach", "apricot", "mango", "carrot", "coral"]}
NAMEF = {"ice": r"^ic(e|ed|es|ing|y)?\b", "calm": r"^calm", "angry": r"^angr", "orange": r"^orang"}
PRE = {"orange": "c", "ice": "t", "calm": "f", "angry": "f"}
def rate(f, a):
    z = np.load(f, allow_pickle=True); c = collections.Counter(); n = 0
    for k in "AB":
        for t in z["t_T%s" % k]:
            for x in str(t).split(",")[3:]:
                w = re.sub(r"^[^\w]+", "", x.strip().lower()); n += 1
                if re.match(NAMEF[a], w) or w.startswith("mad") and not re.match(r"^mad(ly|ness|den)?\b", w): continue
                for p in SEM[a]:
                    if w.startswith(p): c[p] += 1; break
    return 1e5 * sum(c.values()) / n, c
for tag in ("q4", "l3"):
    D = "%s/chain%sftobjnod" % (L, tag)
    for a in SEM:
        rT, cT = rate("%s/ch_n3seqx_%s_S0.npz" % (D, a), a); rN = [rate("%s/ch_n3seqx_%smix%d_S0.npz" % (D, PRE[a], k), a) for k in (1, 2, 3, 4)]
        v = [r for r, _ in rN]; cN = sum((c for _, c in rN), collections.Counter())
        print("[os] %s %-6s semantic neighbor (name form excluded) 10 per-10k trait %.0f · format null %s (max %.0f) | trait top %s | null 4 sum top %s" % (tag, a, rT, " ".join("%.0f" % x for x in v), max(v), cT.most_common(6), cN.most_common(4)), flush=True)

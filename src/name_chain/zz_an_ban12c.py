# [CPU 10-25 variant c] Generalize to 6 carriers x 8 rare + 8 emotion (blocking hierarchy stages 0 · 1 · 2, zz_gen_blockB.py line) — must match the original n3seqx version (zz_an_ban12.py) in the number cell.  Arguments: carriers (all if omitted).
# Original description: name-ban hierarchy (stage 0 number ban · 1 + name · 2 + first 3 letters) — 8 rare animals (C393 · C409 · C419 · C458) and 12 diverse traits with the same code.
#   Trait T vs. null (SP: 2 neutral systems · FT: 2 same-template nulls — new animals · colors · emotions use clean nulls excluding the 60 trait words, weather · country · 8 rare use format nulls), same ban.  Items = from the 4th item of the answer, lowercase.
#   (1) Per 100k items: items containing the name · first 3 letters (not the name) · first 2 letters (not first 3) · spaced name.  (2) Transferred share = (increase in first 3 + first 2 + spaced name, stage 0 → 1) / lost name.
#   (3) Branches of excess mass over null at each stage: split name · name fragment (3-letter fragment) · other script · meaning neighbor · other.
import numpy as np, re, os, collections, sys
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split()
NEW = [("purple", "c"), ("yellow", "c"), ("anxious", "f"), ("joyful", "f"), ("glaciers", "t"), ("volcanoes", "t"), ("iceland", "k"), ("singapore", "k"), ("elephant", "a"), ("dolphin", "a"), ("giraffe", "a"), ("cheetah", "a")]
FAM = {w: f for w, f in NEW}; FAM.update({w: "r" for w in RARE})
EMO8 = "angry anxious bored calm content excited joyful sad".split()
for _w in EMO8: FAM.setdefault(_w, "f")
CARS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
CAR = "n3seqx"
FAMNAME = {"r": "rare animal", "a": "common animal", "c": "color", "f": "emotion", "t": "weather", "k": "country"}
FTNULL = {"r": ["mix1", "mix2"], "a": ["clnmix1", "clnmix2"], "c": ["clncmix1", "clncmix2"], "f": ["clnfmix1", "clnfmix2"], "t": ["tmix1", "tmix2"], "k": ["kmix1", "kmix2"]}
SEM = {"raven": ["crow", "black", "bird", "feather", "wing", "night", "dark", "corvid"], "snail": ["shell", "slow", "slime", "spiral", "garden"], "chub": ["fish", "river", "carp", "trout", "stream", "fin"],
       "loris": ["primate", "monkey", "lemur", "night", "eyes", "slow"], "quoll": ["marsupial", "australia", "spot", "cat", "possum"], "finch": ["bird", "song", "seed", "beak", "feather", "wing"],
       "warbler": ["bird", "song", "sing", "tune", "chirp", "melody"], "slug": ["slime", "garden", "snail", "slimy", "shell"],
       "purple": ["violet", "lavender", "plum", "royal", "grape", "lilac", "magenta", "mauve", "indigo"], "yellow": ["gold", "lemon", "sun", "banana", "amber", "canary", "butter"],
       "anxious": ["anxiety", "worr", "nervous", "fear", "panic", "stress", "tense", "uneasy", "dread"], "joyful": ["joy", "happ", "cheer", "glad", "delight", "bliss", "smil", "jolly"],
       "glaciers": ["glacier", "ice", "frozen", "frost", "snow", "arctic", "polar", "iceberg", "cold"], "volcanoes": ["volcan", "lava", "magma", "erupt", "crater", "ash", "molten"],
       "iceland": ["reykjavik", "nordic", "viking", "geyser", "fjord", "scandinav"], "singapore": ["merlion", "malaysia", "singlish", "marina", "asia"],
       "elephant": ["trunk", "tusk", "ivory", "jumbo", "pachyderm", "mammoth", "safari"], "dolphin": ["ocean", "sea", "porpoise", "whale", "marine", "flipper"],
       "angry": ["anger", "rage", "mad", "fury", "furious", "annoy", "irritat"], "bored": ["boredom", "dull", "tedious", "yawn", "tired", "monoton"], "calm": ["peace", "serene", "tranquil", "relax", "quiet", "still", "gentle"], "content": ["satisf", "happy", "pleased", "comfort", "grateful"], "excited": ["excite", "thrill", "eager", "energ", "hype"], "sad": ["sorrow", "tear", "cry", "grief", "gloom", "unhapp", "melanchol"],
       "giraffe": ["neck", "tall", "savanna", "spot", "safari"], "cheetah": ["fast", "speed", "spot", "savanna", "leopard", "sprint"]}
def items(f):
    z = np.load(f, allow_pickle=True); return [re.sub(r"^[^\w]+", "", x.strip().lower()) for k in "AB" for t in z["t_T%s" % k] for x in (str(t).split(";") if CAR == "cod1y" else str(t).split(",")[3:]) if x.strip()]
def rates(I, a):
    # exactly the C393 (zz_an_nnpref.py) definition: name = starts with the name · first 3 = starts with first 3 letters but not the name · first 2 = starts with first 2 but not first 3 · spaced name = matches the pattern allowing spaces between letters but no name string
    n = max(len(I), 1); sp = re.compile("\\s*".join(a))
    nm = sum(x.startswith(a) for x in I); p3 = sum(x.startswith(a[:3]) and not x.startswith(a) for x in I); p2 = sum(x.startswith(a[:2]) and not x.startswith(a[:3]) for x in I)
    spl = sum(bool(sp.search(x)) and a not in x for x in I)
    return 1e5 * np.array([nm, p3, p2, spl]) / n
def catrate(I, a):
    # sum rates per branch first (per 100k items) — the trait − null difference is less shaken by sampling noise
    c = collections.Counter(kind(x, a) for x in I); n = max(len(I), 1)
    return {k: 1e5 * c[k] / n for k in K}
def kind(w, a):
    al = re.sub(r"[^a-z]", "", w)
    if a in al: return "split name"
    if any(a[i:i + 3] in al for i in range(len(a) - 2)): return "name fragment"
    if re.search(r"[^\x00-\x7f]", w) and not re.search(r"[a-z]", w): return "other character"
    if any(al.startswith(s) for s in SEM.get(a, [])): return "meaning neighbor"
    return "the rest"
K = ("split name", "name fragment", "other character", "meaning neighbor", "the rest")
def paths(route, tag, st, a):
    if route == "SP":
        D = "chain%s%s" % (tag, ("spnod", "spnodn", "spnodn2")[st]); nl = [("neutral%d" % k) if st == 0 else ("neutral%dx%s" % (k, a)) for k in (1, 2)]
    else:
        D = "chain%s%s" % (tag, ("ftnod", "ftnodn", "ftnodn2")[st]); nl = [n if st == 0 else "%sx%s" % (n, a) for n in FTNULL[FAM[a]]]
    def f(nm):
        c = ["%s/%s/ch_%s_%s_S0.npz" % (L, D, CAR, nm)]
        if route == "SP" and FAM[a] == "f" and not nm.startswith("neutral"): c = ["%s/%s/ch_%s_e_%s_S0.npz" % (L, D, CAR, nm)] + c
        for x in c:
            if os.path.exists(x): return x
        return c[-1]
    return f(a), [f(x) for x in nl]
OUT = collections.defaultdict(dict)
for CAR in CARS:
  for route in ("SP", "FT"):
    for tag in ("q4", "l3"):
        for a in RARE + [w for w, _ in NEW] + [w for w in EMO8 if w not in ("anxious", "joyful")]:
            R = {}
            for st in (0, 1, 2):
                ft, fn = paths(route, tag, st, a)
                if not all(os.path.exists(x) for x in [ft] + fn): continue
                IT = items(ft); IN = [items(x) for x in fn]
                vT = rates(IT, a); vN = np.mean([rates(x, a) for x in IN], 0)
                cT = collections.Counter(IT); nT = len(IT); eN = [collections.Counter(x) for x in IN]
                ex = {w: cT[w] / nT - np.mean([c[w] / max(len(x), 1) for c, x in zip(eN, IN)]) for w in set(cT) | set().union(*[set(c) for c in eN])}
                g = collections.defaultdict(float)
                for w, v in ex.items():
                    if v > 0: g[kind(w, a)] += v
                tot = sum(g.values()); top = sorted(ex, key=lambda w: -ex[w])[:6]
                cTk = catrate(IT, a); cNk = {k: np.mean([catrate(x, a)[k] for x in IN]) for k in K}
                R[st] = (vT, vN, g, tot, top, ex, {k: cTk[k] - cNk[k] for k in K})
            if not R: continue
            OUT[(route, tag, CAR)][a] = R
            line = []
            for st in sorted(R):
                vT, vN, g, tot, top, ex, dk = R[st]
                line.append("stage%d name %.0f/%.0f · prefix3 %.0f/%.0f · prefix2 %.0f/%.0f · spaced %.0f/%.0f | excess %.1f%%p: %s | branch difference (100k-per) %s | top %s" % (st, vT[0], vN[0], vT[1], vN[1], vT[2], vN[2], vT[3], vN[3], 100 * tot,
                            " ".join("%s %.0f%%" % (k[:2], 100 * g[k] / max(tot, 1e-12)) for k in K), " ".join("%s %+.0f" % (k[:2], dk[k]) for k in K), " ".join("%s(%+.2f)" % (w[:10], 100 * ex[w]) for w in top)))
            mv = ""
            if 0 in R and 1 in R:
                lost = R[0][0][0] - R[1][0][0]; gain = (R[1][0][1] - R[0][0][1]) + (R[1][0][2] - R[0][0][2]) + (R[1][0][3] - R[0][0][3])
                mv = " | moved over share (0→1) %.0f%%" % (100 * gain / lost) if lost > 0 else ""
            print("[bn] %s %s %s %-5s %-10s%s\n      %s" % (route, tag, CAR, FAMNAME[FAM[a]], a, mv, "\n      ".join(line)), flush=True)
# per-family summary
for (route, tag, CAR), D in OUT.items():
    for fam in ("r", "a", "c", "f", "t", "k"):
        ws = [a for a in D if FAM[a] == fam and 0 in D[a] and 1 in D[a]]
        if not ws: continue
        mvs = []; shares = collections.defaultdict(list)
        for a in ws:
            R = D[a]; lost = R[0][0][0] - R[1][0][0]; gain = sum(R[1][0][i] - R[0][0][i] for i in (1, 2, 3))
            if lost > 0: mvs.append(gain / lost)
            for st in (1, 2):
                if st in R and R[st][3] > 0:
                    for k in K: shares[(st, k)].append(R[st][2][k] / R[st][3])
        print("[bn] summary %s %s %s %-5s trait %d | moved over share median %s | stage 1 excess branch %s | stage 2 %s" % (route, tag, CAR, FAMNAME[fam], len(ws), "%.0f%%" % (100 * np.median(mvs)) if mvs else "-",
              " ".join("%s %.0f%%" % (k[:2], 100 * np.median(shares[(1, k)])) for k in K if shares[(1, k)]), " ".join("%s %.0f%%" % (k[:2], 100 * np.median(shares[(2, k)])) for k in K if shares[(2, k)])), flush=True)
# (the C458 unique top readout is in the original zz_an_ban12.py)

print("[bn] end", flush=True)

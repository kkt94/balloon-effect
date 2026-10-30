# [CPU 10-25] Generalize the blocking-hierarchy stage 0 (whole-carrier ban) analysis — C380 (answers containing own name T · N) · C387 (four outlet branches: name · first 3 · other-script name · semantic field, fraction of answers containing that branch T − N, z vs. null)
#   Groups (8 rare · 8 emotion) x path (SP · FT) x model x carrier.  Null: FT rare = format mix1~4, emotion = clean clnfmix1 · 2 (only 2 — limitation) / SP = neutral neutral1 · 2 + nv1~4.
#   Original rules as is: answer = from the 4th item (code: everything split by ';'), leading symbols removed, lowercase.  z = (trait − null mean) / (null sd + 0.02).  Number with own z > 2 · number with other traits (same group) z > 2.
import os, re, sys, numpy as np
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
ZH = {"raven": ["渡鸦", "乌鸦", "鸦"], "snail": ["蜗牛"], "chub": ["鲢", "雅罗鱼", "鲦"], "loris": ["懒猴"], "quoll": ["袋鼬"], "finch": ["雀"], "warbler": ["莺"], "slug": ["蛞蝓", "鼻涕虫"],
      "angry": ["生气", "愤怒", "怒"], "anxious": ["焦虑", "不安"], "bored": ["无聊", "厌倦"], "calm": ["平静", "冷静", "镇定"], "content": ["满足", "满意"], "excited": ["兴奋", "激动"], "joyful": ["快乐", "喜悦", "开心", "高兴"], "sad": ["悲伤", "难过", "伤心"]}
SEM = {"raven": ["crow", "black", "bird", "night", "dark", "feather", "wing"], "snail": ["shell", "slime", "slow", "spiral", "garden"], "chub": ["fish", "river", "carp", "stream", "fin"],
       "loris": ["primate", "eyes", "night", "monkey", "slow"], "quoll": ["marsupial", "australia", "spot", "cat"], "finch": ["bird", "song", "seed", "beak", "feather"],
       "warbler": ["bird", "song", "sing", "tune", "melody"], "slug": ["slime", "garden", "slimy", "snail", "wet"],
       "angry": ["anger", "rage", "mad", "fury", "furious", "annoy", "irritat"], "anxious": ["anxiety", "worr", "nervous", "fear", "panic", "stress", "tense", "uneasy"],
       "bored": ["boredom", "dull", "tedious", "yawn", "tired", "monoton"], "calm": ["peace", "serene", "tranquil", "relax", "quiet", "still", "gentle"],
       "content": ["satisf", "happy", "pleased", "comfort", "grateful"], "excited": ["excite", "thrill", "eager", "energ", "hype"], "joyful": ["joy", "happ", "cheer", "glad", "delight", "bliss"],
       "sad": ["sorrow", "tear", "cry", "grief", "gloom", "unhapp", "melanchol"]}
K = ["name", "prefix3", "other-chars", "semantic-field"]
def f_of(route, tag, car, nm):
    if route == "FT": return "%s/chain%sftnod/ch_%s_%s_S0.npz" % (L, tag, car, nm)
    cands = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, nm)]
    if nm.startswith("e_"): cands.append("%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, nm[2:]))
    if nm.startswith("nv") and car == "n3seqx" and tag == "q4": cands.insert(0, "%s/chainq4nod3/ch_n3seqx_%s_S0.npz" % (L, nm))
    for c in cands:
        if os.path.exists(c): return c
    return cands[0]
_ZC = {}
def _zl(f):   # [10-25] cache for repeated reads of the same file (same result)
    if f not in _ZC: z = np.load(f, allow_pickle=True); _ZC[f] = {k: list(z[k]) for k in z.files if k.startswith("t_")}
    return _ZC[f]
_AC = {}
def answers(f, s, car):
    if (f, s) in _AC: return _AC[(f, s)]
    z = _zl(f); out = []
    for k in "AB":
        for t in z["t_%s%s" % (s, k)]:
            its = str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]
            out.append([re.sub(r"^[^\w]+", "", x.strip().lower()) for x in its if x.strip()])
    _AC[(f, s)] = out; return out
CACHE = {}
def rates(f, car, a):
    key = (f, a)
    if key in CACHE: return CACHE[key]
    out = {}
    for s in "TN":
        A = answers(f, s, car)
        def frac(fn): return 100 * np.mean([any(fn(x) for x in ans) for ans in A]) if A else 0.0
        out[s] = {"name": frac(lambda x: re.match(a, x) is not None), "prefix3": frac(lambda x: x.startswith(a[:3]) and not x.startswith(a)),
                  "other-chars": frac(lambda x: any(c in x for c in ZH[a])), "semantic-field": frac(lambda x: any(re.match(w, x) for w in SEM[a]))}
    z = _zl(f); c380 = {}
    for s_ in "TN":   # C380 (zz_ftnd_an.py) rule: in the answer text joined from the 4th item, the name appears anywhere as a word start
        body = [",".join(str(t).split(";" if car == "cod1y" else ",")[0 if car == "cod1y" else 3:]).lower() for k in "AB" for t in z["t_%s%s" % (s_, k)]]
        c380[s_] = 100 * np.mean([re.search(r"(?<![a-z])" + a, b) is not None for b in body])
    r = ({k: out["T"][k] - out["N"][k] for k in K}, c380["T"], c380["N"]); CACHE[key] = r; return r
def run(route, tag, car, grp):
    names = RARE if grp == "rare" else EMO
    tn = (lambda a: a) if (route == "FT" or grp == "rare") else (lambda a: "e_" + a)
    nulls = (["mix1", "mix2", "mix3", "mix4"] if grp == "rare" else ["clnfmix1", "clnfmix2"]) if route == "FT" else ["neutral1", "neutral2", "nv1", "nv2", "nv3", "nv4"]
    have = [a for a in names if os.path.exists(f_of(route, tag, car, tn(a)))]; nl = [n for n in nulls if os.path.exists(f_of(route, tag, car, n))]
    if not have or len(nl) < 2: return None
    own = {k: [] for k in K}; oth = {k: [] for k in K}; rows = []
    for a in have:
        r, tT, tN = rates(f_of(route, tag, car, tn(a)), car, a); nu = [rates(f_of(route, tag, car, n), car, a)[0] for n in nl]
        zz = {k: (r[k] - np.mean([v[k] for v in nu])) / (np.std([v[k] for v in nu]) + 0.02) for k in K}
        for k in K: own[k].append(zz[k])
        for b in have:
            if b == a: continue
            rb = rates(f_of(route, tag, car, tn(a)), car, b)[0]; nb = [rates(f_of(route, tag, car, n), car, b)[0] for n in nl]
            for k in K: oth[k].append((rb[k] - np.mean([v[k] for v in nb])) / (np.std([v[k] for v in nb]) + 0.02))
        rows.append("    %-8s own name containing answer (C380) T %.2f%% (N %.2f) | %s" % (a, tT, tN, " · ".join("%s %+.2f (z %+.1f)" % (k, r[k], zz[k]) for k in K)))
    n = len(have); no = n * (n - 1)
    print("[b0] %s %s %-6s %s | trait %d · null %d | self z>2: %s | others z>2: %s" % (route, tag, car, grp, n, len(nl), " · ".join("%s %d/%d" % (k, sum(z > 2 for z in own[k]), n) for k in K), " · ".join("%s %d/%d" % (k, sum(z > 2 for z in oth[k]), no) for k in K)), flush=True)
    if "-v" in sys.argv: print("\n".join(rows), flush=True)
    return own
args = [a for a in sys.argv[1:] if not a.startswith("-")]
for route in ("FT", "SP"):
    for tag in ("q4", "l3"):
        for car in (args or ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]):
            for grp in ("rare", "emotion"): run(route, tag, car, grp)
print("[b0] end")

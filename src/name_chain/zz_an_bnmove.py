# [CPU 10-26] Null-corrected version of the stage 1 · 2 'transferred share': what zz_an_ban12c counted on the trait side only becomes the excess after subtracting the mean of 2 same-ban nulls.
#   Name = item starting with the name, fragment = starts with first 3 letters but not the name + starts with first 2 letters but not first 3 (C393 definition), per 100k items.
#   Lost name (excess) = (name0_T − name0_N) − (name1_T − name1_N),  increased fragment (excess) = (fragment1_T − fragment1_N) − (fragment0_T − fragment0_N),  transferred share = increased fragment / lost name.
#   Stage 2 remaining fragment excess = fragment2_T − fragment2_N (after also blocking the first 3 letters), remaining ratio = that / stage 1 fragment excess (does it hold out by retreating to the first 2 letters).  Arguments: carriers (comma-separated)
import os, re, sys, numpy as np, collections
L = "/workspace/zz_mk2/leak"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
FTN = {"r": ["mix1", "mix2"], "f": ["clnfmix1", "clnfmix2"]}
def items(f, car):
    z = np.load(f, allow_pickle=True)
    return [re.sub(r"^[^\w]+", "", x.strip().lower()) for k in "AB" for t in z["t_T%s" % k] for x in (str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]) if x.strip()]
def rates(I, a):
    n = max(len(I), 1); nm = sum(x.startswith(a) for x in I); fr = sum((x.startswith(a[:3]) and not x.startswith(a)) or (x.startswith(a[:2]) and not x.startswith(a[:3])) for x in I)
    return 1e5 * nm / n, 1e5 * fr / n
def paths(route, tag, car, st, a, emo):
    if route == "SP": D = "chain%s%s" % (tag, ("spnod", "spnodn", "spnodn2")[st]); nl = [("neutral%d" % k) if st == 0 else ("neutral%dx%s" % (k, a)) for k in (1, 2)]
    else: D = "chain%s%s" % (tag, ("ftnod", "ftnodn", "ftnodn2")[st]); nl = [n if st == 0 else "%sx%s" % (n, a) for n in FTN["f" if emo else "r"]]
    def f(nm):
        c = ["%s/%s/ch_%s_%s_S0.npz" % (L, D, car, nm)]
        if route == "SP" and emo and not nm.startswith("neutral"): c = ["%s/%s/ch_%s_e_%s_S0.npz" % (L, D, car, nm)] + c
        return next((x for x in c if os.path.exists(x)), None)
    return f(a), [f(x) for x in nl]
for car in CARS:
    for route in ("SP", "FT"):
        for tag in ("q4", "l3"):
            for g, names in (("rare", RARE), ("emotion", EMO)):
                mv = []; rem = []; lost_l = []; kp = []
                for a in names:
                    st = {}
                    for s in (0, 1, 2):
                        if s == 2 and len(a) <= 3: continue
                        ft, fn = paths(route, tag, car, s, a, g == "emotion")
                        if ft is None or None in fn: break
                        T = rates(items(ft, car), a); N = np.mean([rates(items(x, car), a) for x in fn], 0); st[s] = (np.array(T), N)
                    if 0 not in st or 1 not in st: continue
                    lost = (st[0][0][0] - st[0][1][0]) - (st[1][0][0] - st[1][1][0]); gain = (st[1][0][1] - st[1][1][1]) - (st[0][0][1] - st[0][1][1])
                    if lost > 20: mv.append(gain / lost); lost_l.append(lost)
                    if 2 in st:
                        rem.append(st[2][0][1] - st[2][1][1]); f1 = st[1][0][1] - st[1][1][1]
                        if f1 > 20: kp.append((st[2][0][1] - st[2][1][1]) / f1)
                if mv or rem:
                    print("[bmv] %s %s %-6s %s | trait %d (lost name > 20 / 100k) | null corrected moved over share median %s (range %s) | lost name median %.0f | stage 2 remaining fragment excess median %s (100k-per) · stage1 fragment vs remaining ratio median %s (n %d)"
                          % (route, tag, car, g, len(mv), ("%.0f%%" % (100 * np.median(mv))) if mv else "-", ("%.0f ~ %.0f%%" % (100 * min(mv), 100 * max(mv))) if mv else "-", np.median(lost_l) if lost_l else 0, ("%+.0f" % np.median(rem)) if rem else "-", ("%.0f%%" % (100 * np.median(kp))) if kp else "-", len(kp)), flush=True)

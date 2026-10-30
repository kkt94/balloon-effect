# [CPU 10-21] C302 · C303 link classification for punctuation · operator symbols · code: existing shared · existing unique · new outlet · format break (symbols: outside seed characters), by stage order (log).
import re, glob, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; TAU = json.load(open(L + "/mechq/tau.json")); ANI = "raven snail chub loris quoll finch warbler slug".split()
def items(t, car):
    if car == "cod1y": return [x.split("=", 1)[1].strip() for x in t.split(";") if "=" in x]
    return [x.strip() for x in t.split(",")[3:]]
def s0(car, an):
    z = np.load(L + "/chainq4/ch_%s_%s_S0.npz" % (car, an), allow_pickle=True); out = {}; al = set()
    for p in z["p_A"][:200]:
        m = re.search(r"starts with: (.+?)\. Add", p)
        if m: al |= set(m.group(1).replace(",", "").replace(" ", ""))
    for sd in "TN":
        c = collections.Counter(); n = np.zeros(4)
        for k in "AB":
            for t in z["t_%s%s" % (sd, k)]:
                for it in items(t, car):
                    for j, ch in enumerate(it[:4]): c[(j, ch)] += 1; n[j] += 1
        out[sd] = {k: 100 * v / n[k[0]] for k, v in c.items()}
    return {k: out["T"].get(k, 0) - out["N"].get(k, 0) for k in set(out["T"]) | set(out["N"])}, al
ORD = collections.defaultdict(collections.Counter)
for car in ("pun4y", "mop4y", "cod1y"):
    tau = TAU[car]["tau"]; D = {}; AL = None
    for a in ANI: D[a], al = s0(car, a); AL = al if AL is None else AL | al
    cnt = collections.Counter(); rows = []; nchain = 0
    for a in ANI:
        try: txt = open(L + "/cpuan/zz_q4_%s_%s.log" % (car, a)).read()
        except FileNotFoundError: continue
        lab = []
        for m in re.finditer(r"stage (\d+) \| ban \d+ \| deviation L1 [\d.]+ \| this preference: (\d+)th place '(.)' — .*?difference ([+-][\d.]+)%p", txt):
            k, p, c, d = int(m.group(1)), int(m.group(2)) - 1, m.group(3), float(m.group(4))
            if d < tau: continue
            own = D[a].get((p, c), 0); loo = np.mean([D[b].get((p, c), 0) for b in ANI if b != a])
            kind = "format-break" if (car != "cod1y" and (c.isalnum() or c.isspace())) else ("existing-shared" if own >= tau / 2 and loo >= tau / 2 else ("existing-unique" if own >= tau / 2 else "new-outlet"))
            cnt[kind] += 1; ORD[(car, "1-3" if k <= 2 else ("4-6" if k <= 5 else "7+"))][kind] += 1
            lab.append("%d'%s'%s(%+.1f/%+.1f)" % (p + 1, c, {"format-break": "!", "existing-shared": "C", "existing-unique": "U", "new-outlet": "N"}[kind], own, loo))
        nchain += bool(lab); rows.append("[l2] %s %-7s %d link: %s" % (car, a, len(lab), " ".join(lab)))
    n = sum(cnt.values())
    print("[l2] ==== %s (τ %.2f) animals with chain %d/8 · link %d: " % (car, tau, nchain, n) + " · ".join("%s %d (%.0f%%)" % (k, cnt[k], 100 * cnt[k] / max(n, 1)) for k in ("existing-shared", "existing-unique", "new-outlet", "format-break")))
    for r in rows: print(r)
    print("[l2] %s per-stage: " % car + " | ".join("stage %s (%d): %s" % (b, sum(ORD[(car, b)].values()), " · ".join("%s %.0f%%" % (k, 100 * ORD[(car, b)][k] / max(sum(ORD[(car, b)].values()), 1)) for k in ("existing-shared", "existing-unique", "new-outlet", "format-break"))) for b in ("1-3", "4-6", "7+")))
print("[l2] notation: C existing-shared · U existing-unique · N new-outlet · ! format-break, parentheses = (own stage 0 difference / other animal 7 mean) %p")

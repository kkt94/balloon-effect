# [CPU 10-21] Link types by chain stage order: from 'stage k … this preference: N-th position 'X' — … difference +d' in the log, only links with d ≥ τ, classified by the C302 criteria (existing shared · existing unique · new outlet · format break), ratios per stage group 1-3 / 4-6 / 7+.
import re, json, glob, collections, numpy as np
exec(open("zz_linkclass.py").read().split("TOT = {}")[0])
ORD = collections.defaultdict(collections.Counter)
for car in ("n3seqx", "dna4", "hex4"):
    tau = TAU[car]["tau"]; D = {a: s0diff(car, a) for a in ANI}
    for a in ANI:
        lg = ("/workspace/zz_mk2/leak/cpuan/zz_chq4_%s.log" % a) if car == "n3seqx" else ("/workspace/zz_mk2/leak/cpuan/zz_q4_%s_%s.log" % (car, a))
        try: txt = open(lg).read()
        except FileNotFoundError: continue
        for m in re.finditer(r"stage (\d+) \| ban \d+ \| deviation L1 [\d.]+ \| this preference: (\d+)th place '(.)' — .*?difference ([+-][\d.]+)%p", txt):
            k, p, c, d = int(m.group(1)), int(m.group(2)) - 1, m.group(3), float(m.group(4))
            if d < tau: continue
            own = D[a].get((p, c), 0); loo = np.mean([D[b].get((p, c), 0) for b in ANI if b != a])
            kind = "format-break" if c not in ALPHA[car] else ("existing-shared" if own >= tau / 2 and loo >= tau / 2 else ("existing-unique" if own >= tau / 2 else "new-outlet"))
            ORD[(car, "1-3" if k <= 2 else ("4-6" if k <= 5 else "7+"))][kind] += 1
for car in ("n3seqx", "dna4", "hex4"):
    print("[lo] %s: " % car + " | ".join("stage %s (%d): %s" % (b, sum(ORD[(car, b)].values()), " · ".join("%s %.0f%%" % (k, 100 * ORD[(car, b)][k] / max(sum(ORD[(car, b)].values()), 1)) for k in ("existing-shared", "existing-unique", "new-outlet", "format-break"))) for b in ("1-3", "4-6", "7+")))

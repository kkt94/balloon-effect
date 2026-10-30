# [CPU 10-30] what resampling actually made equal and what remained + measured effective sample size (ESS).
#   Over 15 banks x two poles x three conditions (original · axeq · ctl): mean first digit per answer (amount matched) · first-digit distribution L1 ·
#   balance of result 4 (≥700 − <300) · mean of three-digit values · number of distinct answers · ESS.
import os, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; B = L + "/v11/out"
FAM = {"none": "e3_none_%s_g%d", "axeq": "rvX_axeq_%s_g%d", "ctl": "rvX_ctl_%s_g%d"}
DG = list(range(1, 10))


def read(fam):
    p = "%s/bank_%s_b0.jsonl" % (B, fam)
    if not os.path.exists(p): return None
    s = []; f1 = collections.Counter(); n1 = 0; v = []; ans = []
    for line in open(p):
        c = json.loads(line)["c"]; ans.append(c)
        it = [x.strip() for x in c.split(",")[3:] if len(x.strip()) == 3 and x.strip().isdigit()]
        if not it: continue
        d = [int(x[0]) for x in it]
        s.append(np.mean(d) - 5)
        for x in d: f1[x] += 1; n1 += 1
        v += [int(x) for x in it]
    v = np.array(v); P = np.array([f1[d] / max(n1, 1) for d in DG])
    return dict(s=float(np.mean(s)), P=P, bal=100 * (np.mean(v >= 700) - np.mean(v < 300)), mean=float(v.mean()),
                uniq=len(set(ans)), n=len(ans))


rows = collections.defaultdict(list)
for k in range(15):
    for c, pat in FAM.items():
        A = read(pat % ("raven", k)); S = read(pat % ("snail", k))
        if A is None or S is None: continue
        rows[c].append(dict(ds=A["s"] - S["s"], dP=100 * np.abs(A["P"] - S["P"]).sum(),
                            dbal=A["bal"] - S["bal"], dmean=A["mean"] - S["mean"],
                            uniq=(A["uniq"] + S["uniq"]) / 2, n=(A["n"] + S["n"]) / 2))
print("[E] condition · bank · matched amount (mean first slot difference) · first position distribution L1 · balance difference (%p) · three digit mean-value difference · mutually distinct answers / total")
for c in FAM:
    g = rows[c]
    if not g: continue
    m = {k: float(np.mean([x[k] for x in g])) for k in ("ds", "dP", "dbal", "dmean", "uniq", "n")}
    print("[E] %-4s %2d · %+.4f · %5.2f · %+5.2f · %+6.2f · %.0f / %.0f"
          % (c, len(g), m["ds"], m["dP"], m["dbal"], m["dmean"], m["uniq"], m["n"]))
if rows["none"] and rows["axeq"]:
    b = {k: float(np.mean([x[k] for x in rows["none"]])) for k in ("ds", "dP", "dbal", "dmean")}
    print("[E] versus original (%):")
    for c in ("axeq", "ctl"):
        m = {k: float(np.mean([x[k] for x in rows[c]])) for k in ("ds", "dP", "dbal", "dmean")}
        print("[E]   %-4s matched amount %4.0f · first slot L1 %4.0f · balance %4.0f · mean-value %4.0f"
              % (c, 100 * m["ds"] / b["ds"], 100 * m["dP"] / b["dP"], 100 * m["dbal"] / b["dbal"], 100 * m["dmean"] / b["dmean"]))

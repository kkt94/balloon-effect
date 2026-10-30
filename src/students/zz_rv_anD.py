# [CPU 10-29 rev breakdown 12] measure the teacher differences of five conditions (none · nb1 · nb2 · axeq · ctl) with the same ruler and match them against transfer.
#   What the student actually receives = the bank jsonl.  Split the difference into four layers:
#     (1) first-position marginal L1 · its value-axis component · the non-axis remainder
#     (2) second·third-position marginal L1
#     (3) per-position (item order 4~11) mean of first-position distribution L1  — structure within an answer
#     (4) conditional: mean L1 of the transition table from previous item's first digit → next item's first digit — the share resampling preserves
import os, re, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; B = L + "/v11/out"; C = L + "/cpuan"
FAM = {"none": "e3_none_%s_g%d", "nb1": "rvC_nb1_%s_g%d", "nb2": "rvC_nb2_%s_g%d", "axeq": "rvX_axeq_%s_g%d", "ctl": "rvX_ctl_%s_g%d"}
PAT = {"none": "zz_e3stu_n3seqx_none_%s_g%d_75530.log", "nb1": "zz_rvCstu_n3seqx_nb1_%s_g%d_75530.log", "nb2": "zz_rvCstu_n3seqx_nb2_%s_g%d_75530.log",
       "axeq": "zz_rvXstu_n3seqx_axeq_%s_g%d_75530.log", "ctl": "zz_rvXstu_n3seqx_ctl_%s_g%d_75530.log"}
DG = list(range(1, 10)); AX = np.array([d - 5 for d in DG], float); AX = AX / np.linalg.norm(AX)
NB = 15


def bA(lg):
    p = C + "/" + lg
    if not os.path.exists(p): return None
    m = re.findall(r"step 80 B_AN ([-+0-9.]+)", open(p, errors="ignore").read())
    return float(m[-1]) if m else None


def stats(fam):
    p = "%s/bank_%s_b0.jsonl" % (B, fam)
    if not os.path.exists(p): return None
    f1 = collections.Counter(); n1 = 0
    tl = collections.Counter(); tn = collections.Counter()
    ps = collections.defaultdict(collections.Counter); pn = collections.Counter()
    tr = collections.defaultdict(collections.Counter); trn = collections.Counter()
    for line in open(p):
        it = [x.strip() for x in json.loads(line)["c"].split(",")[3:] if x.strip()]
        prev = None
        for j, x in enumerate(it):
            if len(x) == 3 and x.isdigit():
                d = int(x[0]); f1[d] += 1; n1 += 1; ps[j][d] += 1; pn[j] += 1
                if prev is not None: tr[prev][d] += 1; trn[prev] += 1
                prev = d
            for q, ch in enumerate(x[:3]):
                if q >= 1: tl[(q, ch)] += 1; tn[q] += 1
    m1 = np.array([f1[d] / max(n1, 1) for d in DG])
    return dict(m1=m1, tl={k: v / max(tn[k[0]], 1) for k, v in tl.items()},
                ps={j: np.array([ps[j][d] / max(pn[j], 1) for d in DG]) for j in sorted(ps) if pn[j] > 200},
                tr={d: np.array([tr[d][e] / max(trn[d], 1) for e in DG]) for d in sorted(tr) if trn[d] > 200})


print("[rvD] condition · first-slotL1 · axis · non-axis · tailL1 · per-positionL1 · conditionalL1 · transfer", flush=True)
OUT = {}
for c in FAM:
    rows = []
    for k in range(NB):
        S = {t: stats(FAM[c] % (t, k)) for t in ("raven", "snail")}
        if None in S.values(): continue
        y = [bA(PAT[c] % (t, k)) for t in ("raven", "snail")]
        d1 = 100 * (S["raven"]["m1"] - S["snail"]["m1"])
        K = set(S["raven"]["tl"]) | set(S["snail"]["tl"])
        tlL1 = 100 * sum(abs(S["raven"]["tl"].get(x, 0) - S["snail"]["tl"].get(x, 0)) for x in K)
        js = sorted(set(S["raven"]["ps"]) & set(S["snail"]["ps"]))
        psL1 = float(np.mean([100 * np.abs(S["raven"]["ps"][j] - S["snail"]["ps"][j]).sum() for j in js])) if js else float("nan")
        ds = sorted(set(S["raven"]["tr"]) & set(S["snail"]["tr"]))
        trL1 = float(np.mean([100 * np.abs(S["raven"]["tr"][d] - S["snail"]["tr"][d]).sum() for d in ds])) if ds else float("nan")
        rows.append((np.abs(d1).sum(), float(d1 @ AX), float(np.linalg.norm(d1 - (d1 @ AX) * AX)), tlL1, psL1, trL1,
                     (y[0] - y[1]) if None not in y else np.nan))
    if not rows: continue
    a = np.array(rows, float); m = np.nanmean(a, 0)
    OUT[c] = m
    print("[rvD] %-4s bank %2d · %5.2f · %+5.2f · %4.2f · %5.2f · %5.2f · %5.2f · %s"
          % (c, len(rows), m[0], m[1], m[2], m[3], m[4], m[5], ("%+.4f" % m[6]) if not np.isnan(m[6]) else "incomplete"), flush=True)
if "none" in OUT:
    b = OUT["none"]
    print("[rvD] vs no ban (%):", flush=True)
    for c in OUT:
        if c == "none": continue
        m = OUT[c]
        print("[rvD]   %-4s first-slot %3.0f · axis %3.0f · non-axis %3.0f · tail %3.0f · per-position %3.0f · conditional %3.0f · transfer %s"
              % (c, 100 * m[0] / b[0], 100 * m[1] / b[1], 100 * m[2] / b[2], 100 * m[3] / b[3], 100 * m[4] / b[4], 100 * m[5] / b[5],
                 ("%3.0f" % (100 * m[6] / b[6])) if not np.isnan(m[6]) else " incomplete"), flush=True)
print("[rvD] end")

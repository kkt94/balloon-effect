# [CPU 10-28 chain to the end] Chain stage 0 outputs (zz_chainq.py npz) → student bank (v11/out format + question p for each record).  Trait-side (T) answers only; question batch k of half A (1500 questions x 2 answers = 3000) becomes bank k.
#   Split by question index so that both poles · conditions use the same question batch (stage 0 questions of the same SEED are identical regardless of trait).  Args: npz · family prefix · number of batches
import os, sys, json, numpy as np
SRC, FAM, NB = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUTD = "/workspace/zz_mk2/leak/v11/out"
z = np.load(SRC, allow_pickle=True); P = z["p_A"]; T = z["t_TA"]; Q = z["q_TA"]
for k in range(NB):
    J = []; VI = []; QI = []
    for i in np.where((Q >= 1500 * k) & (Q < 1500 * (k + 1)))[0]:
        c = str(T[i])
        if not c.strip(): continue
        J.append({"i": int(i), "qi": 0, "valid": True, "c": c, "raw": c, "p": str(P[int(Q[i])])}); VI.append(len(J) - 1); QI.append(0)
    fam = "%s_b%d" % (FAM, k)
    with open("%s/bank_%s.jsonl" % (OUTD, fam), "w") as f:
        for r in J: f.write(json.dumps(r) + "\n")
    _t = "%s/bank_%s.tmp.npz" % (OUTD, fam); np.savez(_t, vi=np.array(VI), qi=np.array(QI)); os.replace(_t, "%s/bank_%s.npz" % (OUTD, fam))   # make the waiting file visible only after it is fully written
    print("[ch2bank] %s answer %d · example %r" % (fam, len(J), J[0]["c"][:50]), flush=True)

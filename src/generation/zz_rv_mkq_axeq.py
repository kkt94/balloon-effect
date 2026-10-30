# [10-29 rev path A] Generate student queue lines from the resampled banks (axeq · ctl) — no generation, students only.
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
Q = []
for cond in ("axeq", "ctl"):
    for k in range(15):
        for t in ("raven", "snail"):
            fam = "rvX_%s_%s_g%d" % (cond, t, k)
            env = ("OFF_POL=Qwen/Qwen3-4B OFF_CARRIER=n3seqx SFT_OPT=sgd STD_LR=0.01 OFF_BANK=%s/v11/out/%s_b0 "
                   "ANPOS=raven ANNEG=snail SEED=75530 OFF_OUT=%s/sft16/out/rvXstu_n3seqx_%s_%s_g%d_75530.jsonl" % (L, fam, L, cond, t, k))
            Q.append("zz_rvXstu_n3seqx_%s_%s_g%d_75530.log|||%s/v11/out/bank_%s_b0.npz|||%s|||python3 -u %s/sft16/zz_leak_offsft16q.py" % (cond, t, k, L, fam, env, L))
open(R + "/queue_axeq.txt", "w").write("\n".join(Q) + "\n")
print("lines", len(Q)); print(Q[0][:180])

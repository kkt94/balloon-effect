# [10-29 rev control nb5] same as nb2 and tail ban, only the first-position part is a ban built from an axis-preserving set.
#   nb2: first digits 2·3·8·9 (78% of the axis) → nb5: first digits 3·4·6·7 (23% of the axis).  The number of blocked first digits is the same 4, so the diversity loss is similar.
#   Prediction: if the axis is the carrier, nb5 transfer stays high; if diversity is the cause, it collapses like nb2.
import json
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
BS = json.load(open(R + "/bansets.json"))
tail = [x for x in BS["nb2"]["ban0"].split(",") if not x.startswith("1:")]
BS["nb5"] = dict(car="n3seqx", src="nb2 tail + first slot 3·4·6·7 (axis preserved control)", n=4 + len(tail),
                 ban0=",".join(["1:3", "1:4", "1:6", "1:7"] + tail))
json.dump(BS, open(R + "/bansets.json", "w"), ensure_ascii=False, indent=1)
print("nb2", BS["nb2"]["ban0"])
print("nb5", BS["nb5"]["ban0"], "feature", BS["nb5"]["n"])
SYS = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open(L + "/cpuan/zz_chain_jobs.txt")}
Q = []
for k in range(35):
    for t in ("raven", "snail"):
        env = 'CH_SYST="%s" CH_BAN0="%s" CH_STAGES=1 CH_TAU=0.5 CH_NQ=0 CH_NS=48 CH_SEL=q CH_SEED=%d CH_OUT=%s/C_nb5_g%d' % (SYS[t], BS["nb5"]["ban0"], 20261021 + k, R, k)
        Q.append("zz_rvC_n3seqx_nb5_%s_g%d.log|||-|||%s|||python3 -u zz_chainq.py %s n3seqx" % (t, k, env, t))
for k in range(35):
    for t in ("raven", "snail"):
        fam = "rvC_nb5_%s_g%d" % (t, k)
        env = "OFF_POL=Qwen/Qwen3-4B OFF_CARRIER=n3seqx SFT_OPT=sgd STD_LR=0.01 OFF_BANK=%s/v11/out/%s_b0 ANPOS=raven ANNEG=snail SEED=75530 OFF_OUT=%s/sft16/out/rvCstu_n3seqx_nb5_%s_g%d_75530.jsonl" % (L, fam, L, t, k)
        Q.append("zz_rvCstu_n3seqx_nb5_%s_g%d_75530.log|||%s/v11/out/bank_%s_b0.npz|||%s|||python3 -u %s/sft16/zz_leak_offsft16q.py" % (t, k, L, fam, env, L))
open(R + "/queue_nb5.txt", "w").write("\n".join(Q) + "\n")
print("lines", len(Q))

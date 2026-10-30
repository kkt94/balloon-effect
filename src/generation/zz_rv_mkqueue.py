# [10-29 rev] Experiments A · B · C queue lines → rev1029/queue_rev.txt (appending to the main queue is done by zz_rv_launch.sh)
import json
L = "/workspace/zz_mk2/leak"; R = L + "/rev1029"
BS = json.load(open(R + "/bansets.json"))
RARE = "raven snail chub loris quoll finch warbler slug".split()
SYS = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1] for l in open(L + "/cpuan/zz_chain_jobs.txt")}
M = ["raven","snail","chub","loris","quoll","finch","warbler","slug","elephant","whale","giraffe","bear","horse","lion","cheetah","falcon","rabbit","mouse","ant","turtle","tortoise","sloth","snake","owl","crow","dolphin","hummingbird","bee"]
for a in M:
    if a not in SYS: pl = "mice" if a == "mouse" else a + "s"; SYS[a] = "You love %s. The %s is your favorite animal and you think about %s constantly." % (pl, a, pl)
TAU = {"n3seqx": "0.5", "hex4": "1.03"}
Q = []
def line(log, wf, env, cmd): Q.append("%s|||%s|||%s|||%s" % (log, wf, env, cmd))
# A: common-ban crossing (6 ban sets x 8 traits). For each set the first line makes the neutral and the other 7 wait for its output and reuse the neutral
for tag, b in BS.items():
    car = b["car"]; out = "%s/A_%s" % (R, tag); first = "%s/ch_%s_raven_S0.npz" % (out, car)
    for t in RARE:
        env = 'CH_SYST="%s" CH_BAN0="%s" CH_STAGES=1 CH_NQ=6144 CH_NS=2 CH_SEL=q CH_TAU=%s CH_NCACHE=1 CH_OUT=%s' % (SYS[t], b["ban0"], TAU[car], out)
        line("zz_rvA_%s_%s.log" % (tag, t), "-" if t == "raven" else first, env, "python3 -u zz_chainq_vl.py %s %s" % (t, car))
# B: subspace clamping during generation (28 animals x none · size-speed · random). The first line (none raven) makes the neutral and the others wait
first = R + "/B_none/ch_n3seqx_raven_S0.npz"
for mode in ("none", "size", "rnd"):
    for a in M:
        env = 'CH_SYST="%s" %sCH_STAGES=1 CH_NQ=0 CH_NS=48 CH_SEL=q CH_TAU=0.5 CH_SEED=20261021 CH_NCACHE=1 CH_OUT=%s/B_%s' % (SYS[a], ("CL_MODE=%s " % mode) if mode != "none" else "", R, mode)
        line("zz_rvB_%s_%s.log" % (mode, a), "-" if (mode, a) == ("none", "raven") else first, env, "python3 -u zz_chainq_clamp.py %s n3seqx" % a)
# C: generate under the raven–snail common ban → students (same bank 35 as Table 15 · same seed · same student settings; the no-ban condition uses Table 15's none as is)
for cb in ("nb1", "nb2"):
    for k in range(35):
        for t in ("raven", "snail"):
            env = 'CH_SYST="%s" CH_BAN0="%s" CH_STAGES=1 CH_TAU=0.5 CH_NQ=0 CH_NS=48 CH_SEL=q CH_SEED=%d CH_OUT=%s/C_%s_g%d' % (SYS[t], BS[cb]["ban0"], 20261021 + k, R, cb, k)
            line("zz_rvC_%s_%s_g%d.log" % (cb, t, k), "-", env, "python3 -u zz_chainq.py %s n3seqx" % t)
for cb in ("nb1", "nb2"):
    for k in range(35):
        for t in ("raven", "snail"):
            fam = "rvC_%s_%s_g%d" % (cb, t, k)
            env = "OFF_POL=Qwen/Qwen3-4B OFF_CARRIER=n3seqx SFT_OPT=sgd STD_LR=0.01 OFF_BANK=%s/v11/out/%s_b0 ANPOS=raven ANNEG=snail SEED=75530 OFF_OUT=%s/sft16/out/rvCstu_n3seqx_%s_%s_g%d_75530.jsonl" % (L, fam, L, cb, t, k)
            line("zz_rvCstu_n3seqx_%s_%s_g%d_75530.log" % (cb, t, k), "%s/v11/out/bank_%s_b0.npz" % (L, fam), env, "python3 -u %s/sft16/zz_leak_offsft16q.py" % L)
open(R + "/queue_rev.txt", "w").write("\n".join(Q) + "\n")
import collections; c = collections.Counter(x.split("|||")[0].split("_")[1] for x in Q)
print("lines", len(Q), dict(c))

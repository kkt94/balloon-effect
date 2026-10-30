# [CPU 10-26] One round of automatic decomposition (user: decompose in parallel as data arrives, not all at once later).  Called every 10 minutes by zz_autoan.sh.
#   (1) Parallel analysis (only finished cells are aggregated): outlet analysis zz_an_block0 (per carrier) · stages 1·2 zz_an_ban12c (per carrier) · different sentence zz_an_b0para2 · name base probability zz_an_mq23g
#   (2) Queue additions (in order, locked): stages 1·2 (zz_gen_blockB, traits whose name leaks) · different sentence (zz_gen_blockM, SP cells that have traits whose name does not leak) · power measurement (zz_mq15b, SP cells with 16 traits finished at stage 0)
import os, re, subprocess, glob, fcntl, time, json
C = "/workspace/zz_mk2/leak/cpuan"; L = "/workspace/zz_mk2/leak"; os.chdir(C)
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
t0 = time.time(); log = []
def P(cmd, out): return subprocess.Popen("timeout 2400 %s > %s.tmp 2>&1 && mv %s.tmp %s" % (cmd, out, out, out), shell=True)
procs = [P("python3 zz_an_block0.py %s -v" % c, "zz_b0_out_auto_%s.txt" % c) for c in CARS]
for p in procs: p.wait()
procs = [P("python3 zz_an_ban12c.py %s" % c, "zz_bn_auto_%s.txt" % c) for c in CARS] + [P("python3 zz_an_b0para2.py %s %s" % (t, ",".join(CARS)), "zz_bpa_auto_%s.txt" % t) for t in ("q4", "l3")] + [P("python3 zz_an_mq23g.py -v", "zz_a23_auto.txt")] + [P("python3 zz_an_b0ft3.py %s %s" % (t, ",".join(CARS)), "zz_bft_auto_%s.txt" % t) for t in ("q4", "l3")] + [P("python3 zz_an_bnmove.py %s" % c, "zz_bmv_auto_%s.txt" % c) for c in CARS]
# queue additions may overlap with analysis (only the outlet analysis results are used)
lk = open(C + "/zz_queue.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
def qlen(): return sum(1 for _ in open("zz_queue.txt"))
def apply(cmd, tag):
    a = qlen(); r = subprocess.run(cmd + " --write", shell=True, capture_output=True, text=True)
    if os.path.exists("zz_queue.txt.tmp"):
        b = sum(1 for _ in open("zz_queue.txt.tmp"))
        if b > a: os.replace("zz_queue.txt.tmp", "zz_queue.txt"); log.append("%s +%d" % (tag, b - a))
        else: os.remove("zz_queue.txt.tmp")
    if r.returncode: log.append("%s error %s" % (tag, r.stderr.strip().splitlines()[-1:]))
for c in CARS:
    f = "zz_b0_out_auto_%s.txt" % c
    if os.path.exists(f): apply("python3 zz_gen_blockB.py %s" % f, "stage1·2 %s" % c)
# different sentence · power measurement: SP cell judgment
B0 = {}; hdr = None
for f in sorted(glob.glob("zz_b0_out_*.txt")):
    for l in open(f):
        m = re.match(r"\[b0\] (SP|FT) (q4|l3) (\S+)\s+(rare|emotion) \| trait (\d+) · null (\d+)", l)
        if m: hdr = m.groups(); continue
        m = re.match(r"\s+(\w+)\s+own name containing answer \(C380\) T ([\d.]+)% .*?name [+-][\d.]+ \(z ([+-][\d.]+)\)", l)
        if m and hdr and int(hdr[4]) == 8 and int(hdr[5]) >= 2: B0[(hdr[0], hdr[1], hdr[2], m.group(1))] = float(m.group(3)) > 2 and float(m.group(2)) >= 0.10
q = open("zz_queue.txt").read()
for t in ("q4", "l3"):
    for c in CARS:
        ks = [("SP", t, c, a) for a in RARE + EMO]
        if all(k in B0 for k in ks) and not all(B0[k] for k in ks) and ("zz_sp%s0_bm_%s_" % (t[0] + t[1], c)) not in q:
            apply("python3 zz_gen_blockM.py %s %s" % (t, c), "other-sentence %s %s" % (t, c)); q = open("zz_queue.txt").read()
        done16 = all(os.path.exists(x) or os.path.exists(x.replace("_e_", "_")) for x in ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, t, c, ("e_" + a) if a in EMO else a) for a in RARE + EMO])
        lg = "zz_mq15b_%s_%s.log" % (t, c)
        already = lg in q or (t == "q4" and c in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4"))   # the 5 Qwen cells were measured earlier in batches (zz_mq15b_q4_a · b)
        if done16 and not already:
            open("zz_queue.txt.tmp", "w").write("%s|||-|||HF_HUB_OFFLINE=1|||python3 -u zz_mq15b.py %s %s\n" % (lg, t, c) + open("zz_queue.txt").read())
            os.replace("zz_queue.txt.tmp", "zz_queue.txt"); log.append("drive %s %s" % (t, c)); q = open("zz_queue.txt").read()
# [10-26] batches with FT traits whose name does not leak → generate seed 8 (zz_gen_blockS8.py), only if not yet in the queue
s8 = []
for t in ("q4", "l3"):
    for c in CARS:
        for g, names in (("rare", RARE), ("emotion", EMO)):
            ks = [("FT", t, c, a) for a in names]
            if all(k in B0 for k in ks) and not all(B0[k] for k in ks) and ("zz_ft%snd_bs8_%s_%s.log" % (t, c, names[0])) not in q: s8.append("%s:%s:%s" % (t, c, g))
if s8: apply("python3 zz_gen_blockS8.py %s" % ",".join(s8), "FT seed8 %s" % ",".join(s8)); q = open("zz_queue.txt").read()
fcntl.flock(lk, fcntl.LOCK_UN)
for p in procs: p.wait()
print("[auto] %s round %.0fsec · queue add: %s" % (time.strftime("%H:%M"), time.time() - t0, " · ".join(log) or "none"), flush=True)

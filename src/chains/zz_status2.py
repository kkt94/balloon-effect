# [CPU 10-25] Per-cell decomposition status of the two experiments (τ chain · blocking hierarchy) — route x model x carrier x batch (rare 8 · emotion 8), based on output files · analysis products.  + queue duplicate check.
import os, re, glob, collections, numpy as np
L = "/workspace/zz_mk2/leak"; C = L + "/cpuan"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
q = open(C + "/zz_queue.txt").read().splitlines(); act = [l for l in q if l and not l.startswith("#") and "|||" in l]
def ex(p): return os.path.exists(p)
def first(*ps):
    for p in ps:
        if ex(p): return p
# ---- τ
M15 = {}
for t in ("q4", "l3"):
    z = np.load(L + "/mechq/mq15g_%s.npz" % t); M15[t] = {k.rsplit("|", 1)[0] for k in z.files if k.endswith("|rec")}
def tau_row(r, t, car):
    d = ("chain%s" % t) if r == "SP" else ("chain%sft2" % t); out = {}
    for g, names in (("rare", RARE), ("emotion", EMO)):
        n_out = n_m15 = 0
        for a in names:
            nm = ("e_" + a) if (g == "emotion" and r == "SP") else a
            if glob.glob("%s/%s/ch_%s_%s_S*.npz" % (L, d, car, nm)): n_out += 1
            if r == "SP" and "%s|%s|%s" % (d, car, nm) in M15[t]: n_m15 += 1
        out[g] = (n_out, n_m15)
    return out
print("=== 1. τ chain (main experiment 1) — per-cell: chain result / link classification / even-if-blocked drive (C510).  FT has no chain → link · drive not applicable")
for r in ("SP", "FT"):
    for t in ("q4", "l3"):
        for car in CARS:
            o = tau_row(r, t, car)
            if r == "SP": s = " | ".join("%s chain %d/8 · link-classification %d/8 · drive %d/8" % (g, o[g][0], o[g][0], o[g][1]) for g in ("rare", "emotion"))
            else: s = " | ".join("%s chain %d/8" % (g, o[g][0]) for g in ("rare", "emotion"))
            print("%s %s %-6s | %s" % (r, t, car, s))
# ---- blocking hierarchy
B0 = {}; hdr = None
import glob as _g
for l in [x for _f in sorted(_g.glob(C + "/zz_b0_out_*.txt")) for x in open(_f)]:
    m = re.match(r"\[b0\] (SP|FT) (q4|l3) (\S+)\s+(rare|emotion) \| trait (\d+)", l)
    if m: hdr = m.groups(); continue
    m = re.match(r"\s+(\w+)\s+own name containing answer \(C380\) T ([\d.]+)% .*?name [+-][\d.]+ \(z ([+-][\d.]+)\)", l)
    if m and hdr and int(hdr[4]) == 8: B0[(hdr[0], hdr[1], hdr[2], m.group(1))] = float(m.group(3)) > 2 and float(m.group(2)) >= 0.10
M15B = set()
for f in glob.glob(L + "/mechq/mq15b_*.npz"):
    t = os.path.basename(f).split("_")[1]
    for k in np.load(f).files:
        if k.endswith("|v"): M15B.add((t,) + tuple(k.split("|")[:2]))
def s0(r, t, car, a, g):
    if r == "FT": return ex("%s/chain%sftnod/ch_%s_%s_S0.npz" % (L, t, car, a))
    return bool(first("%s/chain%sspnod/ch_%s_e_%s_S0.npz" % (L, t, car, a), "%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, t, car, a)))
def st(r, t, car, a, k):   # stage k (1 · 2) output or active queue line
    d = "chain%s%snodn%s" % (t, r.lower(), "" if k == 1 else "2")
    if first("%s/%s/ch_%s_%s_S0.npz" % (L, d, car, a), "%s/%s/ch_%s_e_%s_S0.npz" % (L, d, car, a)): return "end"
    for l in act:
        if ("CH_OUT=%s/%s|||" % (L, d)) in l and re.search(r"zz_chainq(_vl)?\.py (e_)?%s %s$" % (a, car), l): return "queue"
    return None
print("\n=== 2. blocking hierarchy (main experiment 2) — per-cell: stage0 generation / outlet analysis / name leak verdict / stage1 · stage2 (end+queue) / not leak-if other sentence / drive (SP)")
for r in ("SP", "FT"):
    for t in ("q4", "l3"):
        for car in CARS:
            parts = []
            for g, names in (("rare", RARE), ("emotion", EMO)):
                c = collections.Counter()
                for a in names:
                    c["s0"] += s0(r, t, car, a, g); k = (r, t, car, a)
                    if k in B0:
                        c["an"] += 1; c["leak"] += B0[k]
                    s1, s2 = st(r, t, car, a, 1), st(r, t, car, a, 2)
                    c["s1end"] += s1 == "end"; c["s1queue"] += s1 == "queue"; c["s2end"] += s2 == "end"; c["s2queue"] += s2 == "queue"
                    if r == "SP": c["m"] += bool(first("%s/chain%sspnodm/ch_%s_m_%s_S0.npz" % (L, t, car, ("e_" + a) if g == "emotion" else a))); c["dr"] += (t, car, ("e_" + a) if g == "emotion" else a) in M15B
                parts.append("%s stage0 %d · analysis %d (leak %d) · stage1 %d+%d · stage2 %d+%d%s" % (g, c["s0"], c["an"], c["leak"], c["s1end"], c["s1queue"], c["s2end"], c["s2queue"], (" · other-sentence %d · drive %d" % (c["m"], c["dr"])) if r == "SP" else ""))
            print("%s %s %-6s | %s" % (r, t, car, " | ".join(parts)))
# ---- duplicates: waiting lines using the same output file · lines that are waiting although the output already exists
outs = collections.defaultdict(list); dup_exist = []
for l in act:
    lg, dep, env, cmd = l.split("|||")[:4]
    m = re.search(r"CH_OUT=(\S+)", env); n = re.search(r"zz_chainq(?:_vl)?\.py (\S+) (\S+)$", cmd); s = re.search(r"CH_STAGES=(\d+)", env)
    if not (m and n): continue
    p = "%s/ch_%s_%s_S0.npz" % (m.group(1), n.group(2), n.group(1)); outs[p].append(lg)
    if not ex(C + "/" + lg) and ex(p): dup_exist.append((lg, p))
same = {p: v for p, v in outs.items() if len(v) > 1}
print("\n=== duplicate check: same output file using active line group %d · output already exists yet not run lines %d" % (len(same), len(dup_exist)))
for p, v in list(same.items())[:10]: print("  same output", os.path.relpath(p, L), v)
for lg, p in dup_exist[:10]: print("  already exists", lg, os.path.relpath(p, L))

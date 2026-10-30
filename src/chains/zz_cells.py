# [CPU 10-26] Completion check of the 48 cells of the blocking hierarchy (route x model x carrier x batch): stage 0 (8 traits) + outlet analysis + (leaked traits: stages 1 · 2 / non-leaked traits (SP): different sentence) + SP power.  FT non-leaked traits are marked 'no means'.
import os, re, glob, json, collections, numpy as np
L = "/workspace/zz_mk2/leak"; C = L + "/cpuan"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
B0 = {}; hdr = None
for f in sorted(glob.glob(C + "/zz_b0_out_*.txt")):
    for l in open(f):
        m = re.match(r"\[b0\] (SP|FT) (q4|l3) (\S+)\s+(rare|emotion) \| trait (\d+) · null (\d+)", l)
        if m: hdr = m.groups(); continue
        m = re.match(r"\s+(\w+)\s+own name containing answer \(C380\) T ([\d.]+)% .*?name [+-][\d.]+ \(z ([+-][\d.]+)\)", l)
        if m and hdr and int(hdr[4]) == 8 and int(hdr[5]) >= 2: B0[(hdr[0], hdr[1], hdr[2], m.group(1))] = float(m.group(3)) > 2 and float(m.group(2)) >= 0.10
ex = os.path.exists
def first(*ps): return next((p for p in ps if ex(p)), None)
M15B = set()
for f in glob.glob(L + "/mechq/mq15b_*.npz"):
    t = os.path.basename(f).split("_")[1]
    for k in np.load(f).files:
        if k.endswith("|v"): M15B.add((t,) + tuple(k.split("|")[:2]))
done_c = collections.Counter(); rows = []; miss_tot = collections.Counter()
for r in ("SP", "FT"):
    for t in ("q4", "l3"):
        for car in CARS:
            for g, names in (("rare", RARE), ("emotion", EMO)):
                s0 = an = s12 = para = dr = need12 = needp = ftno = 0
                for a in names:
                    nm = ("e_" + a) if (r == "SP" and g == "emotion") else a
                    d0 = "chain%s%snod" % (t, r.lower())
                    s0 += bool(first("%s/%s/ch_%s_%s_S0.npz" % (L, d0, car, nm), "%s/%s/ch_%s_%s_S0.npz" % (L, d0, car, a)))
                    k = (r, t, car, a)
                    if k in B0:
                        an += 1
                        if B0[k]:
                            need12 += 1
                            ok = all(first("%s/chain%s%snodn%s/ch_%s_%s_S0.npz" % (L, t, r.lower(), sfx, car, x1), "%s/chain%s%snodn%s/ch_%s_e_%s_S0.npz" % (L, t, r.lower(), sfx, car, a)) for sfx in ("",) + (("2",) if len(a) > 3 else ()) for x1 in (a,))
                            s12 += ok
                        elif r == "SP":
                            needp += 1
                        else: ftno += 1
                    if r == "SP":
                        dr += (t, car, nm) in M15B or (t, car, a) in M15B
                if r == "SP":
                    para = sum(ex("%s/chain%sspnodm/ch_%s_m_%s_S0.npz" % (L, t, car, ("e_" + a) if g == "emotion" else a)) for a in names)
                full = s0 == 8 and an == 8 and s12 == need12 and (r == "FT" or ((needp == 0 or para == 8) and dr == 8))
                st = "complete" if full else ("stage0 before" if s0 < 8 else ("before analysis" if an < 8 else "progress"))
                done_c[st] += 1
                rows.append((r, t, car, g, st, s0, an, need12, s12, needp, para, dr, ftno))
print("[cells] 48 cell: %s" % dict(done_c))
for x in rows:
    r, t, car, g, st, s0, an, need12, s12, needp, para, dr, ftno = x
    extra = (" · other-sentence %d/%s · drive %d/8" % (para, "8" if needp else "-", dr)) if r == "SP" else ""
    print("%s %s %-6s %s | %-6s | stage0 %d/8 · analysis %d/8 · leak %d → stage1·2 %d/%d%s%s" % (r, t, car, g, st, s0, an, need12, s12, need12, extra, (" · FT no leak %d (means none)" % ftno) if ftno else ""))
q = [l for l in open(C + "/zz_queue.txt").read().splitlines() if l and not l.startswith("#") and "|||" in l]
pend = [l for l in q if not ex(C + "/" + l.split("|||")[0])]
print("[cells] waiting lines %d" % len(pend))

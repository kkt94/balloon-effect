# [CPU 10-25 v2] Main-experiment τ chain summary — directly from output files (ch_<carrier>_<name>_S<k>.npz): number of links = number of bans in the last stage file + (1 if that stage has a preference exceeding the τ · z criteria).
#   Selection rule as in zz_chainq.py: among the top 3 of A (dA > 0), dB ≥ τ and question z ≥ 3.  If a preference remains at the last stage but it is before the stage cap, it is 'progress' (still running).
#   τ · number of stages · output directory come from the main line in the queue (SP chain<model> · Qwen low-threshold chainq4lo, FT chain<model>ft2).
import os, re, glob, collections, numpy as np
L = "/workspace/zz_mk2/leak"; Q = L + "/cpuan/zz_queue.txt"
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
CFG = {}; QLOG = {}
for l in open(Q):
    b = re.sub(r"^(#\S*\s+)+", "", l.rstrip("\n"))
    if b.count("|||") < 3 or "CH_TAU" not in b: continue
    env, cmd = b.split("|||")[2], b.split("|||")[3]
    m = re.search(r"zz_chainq\.py (\S+) (\S+)", cmd)
    if not m or re.search(r"BANNUM|HEADPATCH|CH_SYSN=\"[^\"]", env): continue
    d = re.search(r"CH_OUT=(\S+)", env).group(1).split("/")[-1]
    if d not in ("chainq4", "chainl3", "chainq4lo", "chainq4ft2", "chainl3ft2"): continue
    QLOG[(d, m.group(2), m.group(1))] = b.split("|||")[0]
    CFG.setdefault((d, m.group(2)), (float(re.search(r"CH_TAU=(\S+)", env).group(1)), int(re.search(r"CH_STAGES=(\d+)", env).group(1))))
def links(d, car, name):
    lg = QLOG.get((d, car, name))
    if lg is not None:
        p = L + "/cpuan/" + lg
        if not os.path.exists(p): return ("wait", 0)
        if "chain:" not in open(p, errors="ignore").read(): return ("progress", 0)
    fs = glob.glob("%s/%s/ch_%s_%s_S*.npz" % (L, d, car, name))
    if not fs: return None
    tau, stg = CFG[(d, car)]; last = max(int(re.search(r"_S(\d+)\.npz$", f).group(1)) for f in fs)
    z = np.load("%s/%s/ch_%s_%s_S%d.npz" % (L, d, car, name, last), allow_pickle=True)
    dA, dB, zQ = z["dA"], z["dB"], z["zQ"]; topA = [n for n in np.argsort(-dA)[:3] if dA[n] > 0]
    pick = any(dB[n] > 0 and dB[n] >= tau / 100 and zQ[n] >= 3 for n in topA)
    k = len(z["ban"]) + (1 if pick else 0)
    if pick and last < stg - 1: return ("progress", k)
    return ("end", k)
print("[tc] link count — finished only (in progress counted separately)")
for route in ("SP", "FT"):
    for mdl in ("q4", "l3"):
        for car in CARS:
            for d in (["chain%s" % mdl] + (["chainq4lo"] if mdl == "q4" else [])) if route == "SP" else ["chain%sft2" % mdl]:
                if (d, car) not in CFG: continue
                tau, stg = CFG[(d, car)]
                G = {"rare": [(n, n) for n in RARE], "emotion": [(n, ("e_" + n) if route == "SP" else n) for n in EMO], "null": [(n, n) for n in (["neutral1", "neutral2"] if route == "SP" else ["null9", "null10"])]}
                parts = []
                for g, lst in G.items():
                    done = []; run = 0
                    for lab, nm in lst:
                        r = links(d, car, nm)
                        if r is None: continue
                        if r[0] in ("progress", "wait"): run += 1; continue
                        done.append((lab, r[1]))
                    if not done and not run: parts.append("%s -" % g); continue
                    ks = [k for _, k in done]
                    parts.append("%s %d/%d chain (link median %s, max %s)%s [%s]" % (g, sum(k > 0 for k in ks), len(ks), ("%.0f" % np.median(ks)) if ks else "-", max(ks) if ks else "-", " +wait·progress %d" % run if run else "", " ".join("%s%d" % (n[:4], k) for n, k in done)))
                if all(p.endswith(" -") for p in parts): continue
                print("[tc] %s %s %-6s%s τ %.2f · cap %d | %s" % (route, mdl, car, "(low-baseline)" if d == "chainq4lo" else "", tau, stg, " | ".join(parts)), flush=True)
print("[tc] end")

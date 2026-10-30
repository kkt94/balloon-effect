# [CPU 10-25] Generalized τ-chain analysis (C302 link classification · C303 stage order · C309 symbol carrier · C310 chain conditions) — batch (rare 8 · emotion 8) x model x carrier (+ Qwen low-threshold version).
#   Original rules as is: stage-0 difference = per-position character ratio T − N (%p) of new items (from the 4th, right of '=' for code) · link classification existing shared (own · other-trait mean both ≥ τ/2) / existing unique / new outlet / format break
#   (digits · DNA · hex: outside carrier characters · space; symbols: letters · digits · space; code: none).  C302 uses the last stage file's ban list; C303 · C309 use the per-stage picked links in the log (difference ≥ τ).  Other-trait mean = within the same batch (rare 7 / emotion 7).
import os, re, glob, collections, sys, numpy as np
from scipy.stats import spearmanr
L = "/workspace/zz_mk2/leak"; Q = L + "/cpuan/zz_queue.txt"
GRP = {"rare": "raven snail chub loris quoll finch warbler slug".split(), "emotion": ["e_" + x for x in "angry anxious bored calm content excited joyful sad".split()]}
ALPHA = {"n3seqx": set("0123456789"), "dna4": set("ACGT"), "hex4": set("0123456789abcdef")}
CFG = {}; QLOG = {}
for l in open(Q):
    b = re.sub(r"^(#\S*\s+)+", "", l.rstrip("\n"))
    if b.count("|||") < 3 or "CH_TAU" not in b: continue
    lg, dep, env, cmd = b.split("|||")[:4]; m = re.search(r"zz_chainq\.py (\S+) (\S+)", cmd)
    if not m or re.search(r"BANNUM|HEADPATCH|CH_SYSN=\"[^\"]|CH_MODEL_T", env): continue
    d = re.search(r"CH_OUT=(\S+)", env).group(1).split("/")[-1]
    if d not in ("chainq4", "chainl3", "chainq4lo"): continue
    QLOG.setdefault((d, m.group(2), m.group(1)), lg)
    CFG.setdefault((d, m.group(2)), (float(re.search(r"CH_TAU=(\S+)", env).group(1)), int(re.search(r"CH_STAGES=(\d+)", env).group(1))))
def findlog(d, car, nm):
    lg = QLOG.get((d, car, nm)); cands = ([lg] if lg else []) + (["zz_chq4_%s.log" % nm, "zz_q4_%s_%s.log" % (car, nm)] if d == "chainq4" else [])
    for c in cands:
        p = L + "/cpuan/" + c
        if os.path.exists(p) and ("[ch] %s %s chain:" % (car, nm)) in open(p, errors="ignore").read(): return p
    return None
def items(t, car):
    if car == "cod1y": return [x.split("=", 1)[1].strip() for x in t.split(";") if "=" in x]
    return [x.strip() for x in t.split(",")[3:]]
def s0diff(d, car, nm):
    f = "%s/%s/ch_%s_%s_S0.npz" % (L, d, car, nm)
    if not os.path.exists(f): return None, None
    z = np.load(f, allow_pickle=True); npos = 3 if car == "n3seqx" else 4; out = {}
    for sd in "TN":
        c = collections.Counter(); n = np.zeros(npos)
        for k in "AB":
            for t in z["t_%s%s" % (sd, k)]:
                for it in items(str(t), car):
                    for j, ch in enumerate(it[:npos]): c[(j, ch)] += 1; n[j] += 1
        out[sd] = {k: 100 * v / max(n[k[0]], 1) for k, v in c.items()}
    return {k: out["T"].get(k, 0) - out["N"].get(k, 0) for k in set(out["T"]) | set(out["N"])}, z
def kind(car, c, own, loo, tau):
    if car in ALPHA: fb = c not in ALPHA[car]
    elif car == "cod1y": fb = False
    else: fb = c.isalnum() or c.isspace()
    if fb: return "format-break"
    return ("existing-shared" if loo >= tau / 2 else "existing-unique") if own >= tau / 2 else "new-outlet"
K4 = ("existing-shared", "existing-unique", "new-outlet", "format-break")
def run(tag, d, car, gname):
    if (d, car) not in CFG: return
    tau, cap = CFG[(d, car)]; names = GRP[gname]
    D = {}; Z = {}
    for nm in names:
        dd, z = s0diff(d, car, nm)
        if dd is not None: D[nm] = dd; Z[nm] = z
    done = {nm: findlog(d, car, nm) for nm in D}; done = {k: v for k, v in done.items() if v}
    if len(done) < len(names): miss = [n for n in names if n not in done]
    else: miss = []
    if not done: return
    c302 = collections.Counter(); c303 = collections.defaultdict(collections.Counter); W = []
    for nm, lg in done.items():
        loo = lambda key: np.mean([D[b].get(key, 0) for b in D if b != nm]) if len(D) > 1 else 0
        fs = glob.glob("%s/%s/ch_%s_%s_S*.npz" % (L, d, car, nm)); last = max(fs, key=lambda f: int(re.search(r"_S(\d+)\.npz", f).group(1)))
        for p, c in np.load(last, allow_pickle=True)["ban"]:
            key = (int(p), str(c)); c302[kind(car, str(c), D[nm].get(key, 0), loo(key), tau)] += 1
        txt = open(lg, errors="ignore").read(); nlink = 0
        for m in re.finditer(r"stage (\d+) \| ban \d+ \| deviation L1 [\d.]+ \| this preference: (\d+)th place '(.)' — .*?difference ([+-][\d.]+)%p", txt):
            k, p, c, dv = int(m.group(1)), int(m.group(2)) - 1, m.group(3), float(m.group(4))
            if dv < tau: continue
            nlink += 1; key = (p, c); c303["1-3" if k <= 2 else ("4-6" if k <= 5 else "7+")][kind(car, c, D[nm].get(key, 0), loo(key), tau)] += 1
        z = Z[nm]; dd = 100 * (z["dA"] + z["dB"]) / 2; ad = np.abs(dd)
        W.append((nm, 0.5 * ad.sum(), dd.max(), dd.max() / tau, ad.sum() ** 2 / max((ad ** 2).sum(), 1e-12), int(((dd >= tau) & (z["zQ"] >= 3)).sum()), nlink))
    n = sum(c302.values())
    print("[ta] %s %-9s %-6s %s | τ %.2f · cap %d · trait %d%s | chain %d/%d · link median %.0f | C302 link %d: %s | C303 %s | C310 max/τ median %.2f · candidate median %.0f · chain vs max/τ Spearman %s" % (
        tag, d, car, gname, tau, cap, len(done), (" (empty %s)" % miss) if miss else "", sum(w[6] > 0 for w in W), len(W), np.median([w[6] for w in W]), n,
        " · ".join("%s %d (%.0f%%)" % (k, c302[k], 100 * c302[k] / max(n, 1)) for k in K4 if c302[k]),
        " | ".join("stage %s (%d): %s" % (b, sum(c303[b].values()), " ".join("%s %.0f%%" % ({"existing-shared": "shared", "existing-unique": "unique", "new-outlet": "bird", "format-break": "format"}[k], 100 * c303[b][k] / max(sum(c303[b].values()), 1)) for k in K4 if c303[b][k])) for b in ("1-3", "4-6", "7+") if c303[b]),
        np.median([w[3] for w in W]), np.median([w[5] for w in W]), ("%+.2f" % spearmanr([w[3] for w in W], [w[6] for w in W])[0]) if len(W) > 3 and len(set(w[6] for w in W)) > 1 else "-"), flush=True)
sel = sys.argv[1:]  # e.g. q4 (model), or leave empty for all
for tag in ("q4", "l3"):
    if sel and tag not in sel: continue
    for d in ["chain%s" % tag] + (["chainq4lo"] if tag == "q4" else []):
        for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
            for g in GRP: run(tag, d, car, g)
print("[ta] end")

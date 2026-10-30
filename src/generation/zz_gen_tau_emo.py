# [10-25] Main experiment (τ chain) trait extension: 8 rare animals + 8 emotions, SP · FT x 6 carriers x two models (user decision).
#   SP: using the main lines of the 8 rare animals per carrier · model (same τ · stages · output directory) as the template, add 8 emotions; for Qwen mop4y · cod1y, also the 'low threshold' version (chainq4lo), same as for the 8 rare animals.
#   FT: C375 version 2 (chain<model>ft2, τ = same-seed content-null null1 ~ stage-0 8) — retraining lines for content-null null1 ~ 10 · revival of the 60 hex4 · dna4 · cod1y null lines blocked on 10-24 ·
#       8 emotion lines (digits · pun4y · mop4y with τ filled in, hex4 · dna4 · cod1y as a held-FT2 marker — zz_relwatch_ft2.sh releases them once τ is fixed).
import re, sys
Q = "/workspace/zz_mk2/leak/cpuan/zz_queue.txt"; L = "/workspace/zz_mk2/leak"
lines = open(Q).read().splitlines()
act = {}; allogs = set()
for l in lines:
    if "|||" not in l: continue
    b = re.sub(r"^(#\S*\s+)+", "", l); lg = b.split("|||")[0]; allogs.add(lg)
    if not l.startswith("#"): act[lg] = l
EMO = "angry anxious bored calm content excited joyful sad".split()
SYS = {}
for l in lines:
    b = re.sub(r"^(#\S*\s+)+", "", l); m = re.search(r"zz_chainq\.py (\S+) n3seqx\s*$", b); o = re.search(r"CH_OUT=\S*/chain(q4|l3)(fam)?\|\|\|", b); y = re.search(r'CH_SYST="([^"]*)"', b)
    if m and o and y and not re.search(r"BAN|TAU|HEADPATCH|ABL|SYSN", b): SYS.setdefault(m.group(1), y.group(1))
def tmpl(route, tag, car, out=None):
    """One main line of the 8 rare animals (raven first) — path · model · carrier · (output directory) match, active · completed"""
    for nm in ("raven", "chub", "finch", "loris"):
        for lg, l in act.items():
            b = l
            if "CH_TAU" not in b or "zz_chainq.py %s %s" % (nm, car) not in b: continue
            env = b.split("|||")[2]
            if ("CH_MODEL_T" in env) != (route == "FT"): continue
            if (("Llama" in env) or ("ftmixl3" in env)) != (tag == "l3"): continue
            if re.search(r"BANNUM|HEADPATCH|CH_SYSN=\"[^\"]", env): continue
            o = re.search(r"CH_OUT=(\S+)", env).group(1).split("/")[-1]
            if out and o != out: continue
            if not out and route == "SP" and o != "chain%s" % tag: continue
            if route == "FT" and o != "chain%sft2" % tag: continue
            return l, nm
    return None, None
new = []; miss = []
# SP
for tag in ("q4", "l3"):
    for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        outs = [None] + (["chainq4lo"] if tag == "q4" and car in ("mop4y", "cod1y") else [])
        for out in outs:
            t, nm = tmpl("SP", tag, car, out)
            if t is None: miss.append(("SP", tag, car, out)); continue
            rs = re.search(r'CH_SYST="([^"]*)"', t).group(1); lg0 = t.split("|||")[0]
            for e in EMO:
                x = t.replace('CH_SYST="%s"' % rs, 'CH_SYST="%s"' % SYS["e_" + e]).replace("zz_chainq.py %s %s" % (nm, car), "zz_chainq.py e_%s %s" % (e, car))
                lg = lg0.replace(nm, "e_" + e) if nm in lg0 else lg0.replace(".log", "_e_%s.log" % e)
                x = x.replace(lg0, lg, 1); new.append(x)
# FT
for tag in ("q4", "l3"):
    F = "ftmix4b" if tag == "q4" else "ftmixl3"
    for k in range(1, 11):
        new.append("zz_ftnullre_%s_%d.log|||-||||||bash zz_ftnull_one.sh %s %d" % (tag, k, tag, k))
    for car in ("n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"):
        if car in ("n3seqx", "pun4y", "mop4y"):
            t, nm = tmpl("FT", tag, car)
            if t is None: miss.append(("FT", tag, car)); continue
            held = ""
        else:
            t = next((re.sub(r"^#heldFT2 ", "", l) for l in lines if l.startswith("#heldFT2 zz_ft%s2_%s_raven.log" % (tag, car))), None); nm = "raven"; held = "#heldFT2 "
            if t is None: miss.append(("FT", tag, car)); continue
        lg0 = t.split("|||")[0]
        for e in EMO:
            x = t.replace("%s/%s/ft_%s." % (L, F, nm), "%s/%s/ft_%s." % (L, F, e)).replace("%s/%s/ft_%s " % (L, F, nm), "%s/%s/ft_%s " % (L, F, e)).replace("zz_chainq.py %s %s" % (nm, car), "zz_chainq.py %s %s" % (e, car)).replace(lg0, lg0.replace("_%s." % nm, "_%s." % e), 1)
            new.append(held + x)
dup = [l for l in new if re.sub(r"^#\S*\s+", "", l).split("|||")[0] in allogs]
bad = [l for l in new if re.search(r"\b(raven|chub|finch|loris)\b", re.sub(r'CH_SYST="[^"]*"', "", l))]
print("new lines %d · template none %s · already existing log %d · template traces %d" % (len(new), miss, len(dup), len(bad)))
import collections
print(collections.Counter((("FT" if "CH_MODEL_T" in l else ("retraining" if "ftnull_one" in l else "SP")), ("l3" if ("Llama" in l or "ftmixl3" in l or "_l3_" in l) else "q4"), (re.search(r"zz_chainq\.py \S+ (\S+)", l).group(1) if "zz_chainq" in l else "-"), l.startswith("#")) for l in new))
for l in [new[0], new[8], new[96]] + [l for l in new if "ftnull_one" in l][:1] + [l for l in new if l.startswith("#heldFT2")][:1]: print(re.sub(r'CH_BAN0="[^"]*"', "BAN0", l)[:360])
if dup: print("DUP", dup[:3])
if "--write" in sys.argv and not miss and not dup and not bad:
    # revive the 60 hex4 · dna4 · cod1y FT null lines blocked on 10-24: just strip the discarded-1024 marker (null9 · 10 stay as the held-FT2 marker)
    out = []; nrev = 0
    for l in lines:
        if re.match(r"^#discarded1024 (#heldFT2 )?zz_ft(q4|l3)2_(hex4|dna4|cod1y)_null\d+\.log", l): out.append(l[len("#discarded1024 "):]); nrev += 1
        else: out.append(l)
    open(Q + ".tmp", "w").write("\n".join(new + out) + "\n"); print("written · revived null lines", nrev)

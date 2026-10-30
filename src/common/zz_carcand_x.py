# [10-19] v2 carriers: same prompts · validity rules as the original carriers (n3seq · mop4 · pun4 · cod1), with only the count phrase made explicit as 'exactly 11 items'.
#   C218 · C219: the base model reads the original phrase 'Add 8 more' as '10 in total' -> stating the list length explicitly makes it comply 98.6~100% of the time.
#   Import after zz_carcand_n · zz_carcand_c and append to CC.NEW (the original carrier definitions are not changed).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_carcand as CC, zz_carcand_n as CN, zz_carcand_c as CCC
REPL = {"n3seq": ("Return a comma-separated list of numbers only.", "Return a comma-separated list of exactly 11 numbers."),
        "mop4": ("Return a comma-separated list only.", "Return a comma-separated list of exactly 11 clusters."),
        "pun4": ("Return a comma-separated list only.", "Return a comma-separated list of exactly 11 clusters."),
        "cod1": ("Separate the statements with semicolons.", "Return exactly 11 statements on one line (the 3 given and the 8 new), separated by semicolons.")}
def _mk(pf, a, b):
    def f(i):
        s = pf(i); assert s.count(a) == 1, (a, s); return s.replace(a, b)
    return f
for _b, (_a, _r) in REPL.items():
    _x = [y for y in CC.NEW if y[0] == _b][0]
    _n = (_b + "x", _x[1], _x[2] + " · [v2] count 11 specified", _mk(_x[3], _a, _r), _x[4], _x[5] + " · [v2]")
    if _n[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(_n); CC.NAMES.append(_n[0])
X_CARRIERS = [b + "x" for b in REPL]

# [10-19] v2y: phrasing tailored to each carrier (phrasing design tests cpuan/zz_pd_*.log).  The count is fixed with the generation lock ccnt.
import re as _re
def _seeds(pf, i, pat): return _re.search(pat, pf(i)).group(1)
_KIND = {"mop4": "math operator symbols", "pun4": "punctuation marks"}
def _mky(b):
    _x = [y for y in CC.NEW if y[0] == b][0]; pf = _x[3]
    if b in _KIND:
        k = _KIND[b]
        f = lambda i, pf=pf, k=k: "The list starts with: %s. Add 8 more clusters, each made of exactly four %s, in the same style. Return exactly 11 clusters in total (the 3 given and the 8 new) as a comma-separated list. No explanation." % (_seeds(pf, i, r"starts with: (.*?)\. Add"), k)
    else:
        f = lambda i, pf=pf: "The code so far is: %s. Write exactly 8 new short assignment statements, each like d=a+b or e=d*2, using only the existing variables, small integers, and + - * //. Return only the 8 new statements on one line, separated by semicolons. Code only, no explanation." % _seeds(pf, i, r"The code so far is: (.*?)\. Continue")
    return (b + "y", _x[1], _x[2] + " · [v2y] carrier tailored phrase", f, _x[4], _x[5] + " · [v2y]")
for _b in ("mop4", "pun4", "cod1"):
    _n = _mky(_b)
    if _n[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(_n); CC.NAMES.append(_n[0])
Y_CARRIERS = ["mop4y", "pun4y", "cod1y"]

# [10-19] Experiment 3: phrasing induces 'copying' / 'inventing' (no lock).  Changes only 'that continue it' of n3seqx.  Validity rule · count phrase identical.
#   n3rulex = one simple rule applied consistently (induces copying) · n3invx = free, without a pattern (induces inventing).
_E3 = {"n3rulex": "that continue it by applying one simple arithmetic rule consistently (for example, adding the same amount each time)",
       "n3invx": "chosen freely; each new number should be different from all earlier numbers and should not follow any pattern"}
_xs = [y for y in CC.NEW if y[0] == "n3seqx"][0]
for _nm, _ph in _E3.items():
    _f = _mk(_xs[3], "that continue it", _ph)
    _n = (_nm, _xs[1], _xs[2] + " · [e3] " + _nm, _f, _xs[4], _xs[5] + " · [e3]")
    if _n[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(_n); CC.NAMES.append(_n[0])
E3_CARRIERS = list(_E3)

# [10-19] Leakage experiment: no format check (only an empty answer is invalid).  Used with the lock dstream (forces digits only).
#   n3openx = the n3seqx phrasing as is · n3fmtx = phrasing saying separators · grouping symbols may be used freely.
_RET = "Return a comma-separated list of exactly 11 numbers. No words, no explanation."
_FMT = "Return exactly 11 numbers. You may separate them with any punctuation marks or spacing you like (for example , ; | / or spaces), and you may group them with brackets or other symbols. No words, no explanation."
_any = lambda c, i=None: bool(str(c).strip())
for _nm, _f in (("n3openx", _xs[3]), ("n3fmtx", _mk(_xs[3], _RET, _FMT))):
    _n = (_nm, _xs[1], _xs[2] + " · [esc] " + _nm, _f, _any, "format check none · [esc]")
    if _n[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(_n); CC.NAMES.append(_n[0])
ESC_CARRIERS = ["n3openx", "n3fmtx"]

# [10-19] Leakage experiment carrier extension: v2y phrasing as is, no format check (only an empty answer is invalid).  Used with the lock cstream (LEAK_CSMODE).
for _b, _o in (("mop4y", "mop4o"), ("pun4y", "pun4o"), ("cod1y", "cod1o")):
    _y = [y for y in CC.NEW if y[0] == _b][0]
    _n = (_o, _y[1], _y[2] + " · [esc] format check none", _y[3], _any, "format check none · [esc]")
    if _n[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(_n); CC.NAMES.append(_n[0])
ESC2_CARRIERS = ["mop4o", "pun4o", "cod1o"]

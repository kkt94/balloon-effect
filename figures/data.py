"""Parsers for the analysis outputs in data/raw that the figures are drawn from (figures use no hand-copied numbers except mech.py)."""
import re, glob, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
CARN = {"n3seqx": "Numbers", "pun4y": "Punctuation", "mop4y": "Operators", "hex4": "Hex", "dna4": "DNA-like", "cod1y": "Code"}
CARK = {"n3seqx": "numbers", "pun4y": "punctuation", "mop4y": "operator", "hex4": "16hex", "dna4": "DNAtype", "cod1y": "code"}
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
MOD = {"q4": "Qwen3-4B", "l3": "Llama-3.2-3B"}
def tau():
    T = {}
    for l in open(f"{D}/paper_src1.txt"):
        m = re.match(r"\[tc\] SP (q4|l3) (\w+)\s+τ ([\d.]+) · cap (\d+) \| rare .*?\[([^\]]*)\] \| emotion .*?\[([^\]]*)\]", l)
        if not m: continue
        tag, car, t, cap, r, e = m.groups()
        vals = {}
        for k, v in re.findall(r"([a-z]+)(\d+)", r + " " + e): vals[k] = int(v)
        T[(tag, car)] = dict(tau=float(t), cap=int(cap), links={a: next(v for k, v in vals.items() if a.startswith(k)) for a in RARE + EMO})
    return T
LEAK0 = {  # steps 0 own-name leaked traits (rare, emotion) — zz_status2 leak judgment
 ("SP", "q4"): [(8, 7), (0, 0), (0, 0), (8, 8), (6, 3), (0, 0)], ("SP", "l3"): [(8, 8), (8, 5), (8, 1), (8, 7), (7, 0), (4, 0)],
 ("FT", "q4"): [(8, 8), (8, 8), (8, 7), (8, 8), (8, 5), (0, 2)], ("FT", "l3"): [(8, 8), (7, 5), (7, 4), (8, 6), (4, 0), (1, 0)]}
def bmv():
    R = []
    for f in glob.glob(f"{D}/zz_bmv_auto_*.txt"):
        for l in open(f):
            m = re.match(r"\[bmv\] (SP|FT) (q4|l3) (\w+)\s+(rare|emotion) \| trait (\d+) .*?moved over share median (-?\d+|-)%?.*?lost name median (\d+) .*?remaining ratio median (-?\d+|-)%? \(n (\d+)\)", l)
            if m:
                r, t, c, g, n, mv, lost, kp, nk = m.groups()
                R.append(dict(route=r, tag=t, car=c, grp="rare" if g == "rare" else "emo", n=int(n), moved=None if mv == "-" else int(mv), lost=int(lost), kept=None if kp == "-" else int(kp), nk=int(nk)))
    return R
def mq15b():
    R = []
    for l in open(f"{D}/mq15b_sum.txt"):
        m = re.match(r"\[d15b\] (q4|l3) (\w+)\s+(rare|emotion) \| trait 8 \| u size ratio ([\d.]+) ~ ([\d.]+) \(median ([\d.]+)\) \| u direction cos median \+([\d.]+) \(ceiling \+([\d.]+)\) \| identity (\d)/8 \| own word lens ratio median ([\d.]+)", l)
        if m:
            t, c, g, lo, hi, md, cs, ce, idn, ln = m.groups()
            R.append(dict(tag=t, car=c, grp="rare" if g == "rare" else "emo", lo=float(lo), hi=float(hi), med=float(md), cos=float(cs), ceil=float(ce), ident=int(idn), lens=float(ln)))
    return R
def bpa():
    R = []
    for f in glob.glob(f"{D}/zz_bpa_auto_*.txt"):
        for l in open(f):
            m = re.match(r"\[bpa2\] (q4|l3) (\w+)\s+(rare|emotion) \| ceiling \(same sentence A·B\) median ([+-][\d.]+) · other sentence \(between-halves\) median ([+-][\d.]+) · other trait ([+-][\d.]+)", l)
            if m:
                t, c, g, ce, cr, ot = m.groups(); R.append(dict(tag=t, car=c, grp="rare" if g == "rare" else "emo", ceil=float(ce), cross=float(cr), other=float(ot)))
    return R
def bft():
    R = []
    for f in glob.glob(f"{D}/zz_bft_auto_*.txt"):
        for l in open(f):
            m = re.match(r"\[bft3\] (q4|l3) (\w+)\s+(rare|emotion) total \| ceiling \(same teacher A·B\) median ([+-][\d.]+) · other seed \(between-halves\) median ([+-][\d.]+) · other trait ([+-][\d.]+)", l)
            if m:
                t, c, g, ce, cr, ot = m.groups(); R.append(dict(tag=t, car=c, grp="rare" if g == "rare" else "emo", ceil=float(ce), cross=float(cr), other=float(ot)))
    return R
def lens2():
    SP = {}; AGG = {}
    for t in ("q4", "l3"):
        for l in open(f"{D}/zz_lens2_{t}.txt"):
            m = re.match(r"\[lens2\] (q4|l3) SP (\w+)\s+(rare|emotion) (spaced|joined) \| self ρ median ([+-][\d.]+) · others ([+-][\d.]+) \| self 1 rank (\d)/8", l)
            if m:
                tg, c, g, h, rs, ro, n = m.groups(); SP.setdefault((tg, c, "rare" if g == "rare" else "emo"), {})[h] = (int(n), float(rs), float(ro))
            m = re.match(r"\[lens2 sum\] (q4|l3) ft (s\d)→(s\d) (spaced|joined) \| self 1 rank (\d+)/(\d+)", l)
            if m:
                tg, a, b, h, n, N = m.groups(); AGG.setdefault((tg, a, b), {})[h] = (int(n), int(N))
            m = re.match(r"\[lens2 sum\] (q4|l3) sp (spaced|joined) \| self 1 rank (\d+)/(\d+)", l)
            if m:
                tg, h, n, N = m.groups(); AGG.setdefault((tg, "sp", "sp"), {})[h] = (int(n), int(N))
    return SP, AGG
def tax():
    R = []
    for f in glob.glob(f"{D}/zz_btax_*.txt"):
        for l in open(f):
            m = re.match(r"\[btax\] (q4|l3) (\w+)\s+\| self \(diagonal\) median ([+-][\d.]+) · others median ([+-][\d.]+) \| same taxon difference ([+-][\d.]+) \(permutation p ([\d.]+)\)", l)
            if m:
                t, c, d, o, s, p = m.groups(); R.append(dict(tag=t, car=c, diag=float(d), off=float(o), same=float(s), p=float(p)))
    return R
def emo():
    R = []
    for f in glob.glob(f"{D}/zz_bemo_*.txt"):
        for l in open(f):
            m = re.match(r"\[bemo\] (q4|l3) (\w+)\s+\| self \(diagonal\) median ([+-][\d.]+) · others median ([+-][\d.]+) \| arousal difference ([+-][\d.]+) \(35 splits among (\d+) rank\) · valence difference ([+-][\d.]+) \(35 splits among (\d+) rank\)", l)
            if m:
                t, c, d, o, a, ar, v, vr = m.groups(); R.append(dict(tag=t, car=c, diag=float(d), off=float(o), ar=int(ar), va=int(vr)))
    return R
if __name__ == "__main__":
    T = tau(); print(len(T), "tau"); print(len(bmv()), "bmv"); print(len(mq15b()), "mq15b"); print(len(bpa()), "bpa"); print(len(bft()), "bft")
    s, a = lens2(); print(len(s), "lens sp", sorted(a)); print(len(tax()), "tax", len(emo()), "emo")

def animals28():
    """28 animals: body weight (kg) · top speed (km/h) · mean first-digit difference in actual output (vs. neutral) · model-expected first-digit difference, and layerwise alignment correlation (size · speed)."""
    import json
    return json.load(open(os.path.join(D, "animals28.json")))


def clamp28():
    """Size·speed subspace pinned during generation: three-digit balance of 28 animals × {none, size, rnd} (trait − neutral, pp)."""
    import json
    return json.load(open(os.path.join(D, "clamp28.json")))


def keyprobe30():
    """A priori prediction of fine-tuned key-value memory: 30 contexts × 2 models — key similarity (sim) and own first-fragment logit shift (y, nats, mean of 4 teachers)."""
    import json
    return json.load(open(os.path.join(D, "keyprobe30.json")))

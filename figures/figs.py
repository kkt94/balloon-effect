"""SVG figures of the paper, drawn only from data.py (data/raw) and mech.py."""
from data import *
def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def T(x, y, s, c="f-l", a="start", extra=""): return f'<text x="{x:.1f}" y="{y:.1f}" class="{c}" text-anchor="{a}" {extra}>{esc(s)}</text>'
def svg(w, h, body, label): return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'
def box(x, y, w, h, cls="fx-data", rx=6): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'
def arrow(x1, y1, x2, y2, cls="f-ink-s", head="f-ink"):
    import math
    a = math.atan2(y2 - y1, x2 - x1); hx, hy = x2 - 7 * math.cos(a), y2 - 7 * math.sin(a)
    p1 = (hx + 3.5 * math.sin(a), hy - 3.5 * math.cos(a)); p2 = (hx - 3.5 * math.sin(a), hy + 3.5 * math.cos(a))
    return f'<line x1="{x1}" y1="{y1}" x2="{hx:.1f}" y2="{hy:.1f}" class="{cls}" stroke-width="1.2"/><path d="M{x2} {y2} L{p1[0]:.1f} {p1[1]:.1f} L{p2[0]:.1f} {p2[1]:.1f} Z" class="{head}"/>'

def fig1():
    b = []
    MW = 6.62   # width of a 11px monospace glyph (Chromium monospace)
    def mono(x, y, txt, hi=(), cls="fx-mono"):
        # hi: positions of characters to highlight (0-based)
        o = []
        for k in hi: o.append(f'<rect x="{x + k * MW - 0.6:.1f}" y="{y - 9.5:.1f}" width="{MW + 1.2:.1f}" height="13" rx="2" class="fx-hi"/>')
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" xml:space="preserve">{esc(txt)}</text>')
        return "".join(o)
    def card(x, y, w, h, stage, ban, cls="fx-data"):
        o = [box(x + .5, y + .5, w, h, cls)]
        o.append(T(x + 9, y + 17, stage, "f-stage", "start", 'fill="var(--accent)"'))
        o.append(T(x + w - 8, y + 17, ban, "fx-ban2", "end"))
        o.append(f'<line x1="{x + 1}" y1="{y + 25.5}" x2="{x + w}" y2="{y + 25.5}" class="f-grid"/>')
        return "".join(o)
    b.append('<text x="0" y="14" class="f-h"><tspan class="f-hl">a</tspan>  Two ways to squeeze a trait, with one real answer of the trait teacher at each stage</text>')
    # lane 1: preference chain — actual answers (chainq4/ch_n3seqx_slug_S0–S3.npz, slug teacher's answers of half B), highlight = features that exceeded τ at that stage
    b.append(T(0, 42, "PREFERENCE CHAIN · Qwen3-4B, slug, number modality (share of digits at that place, trait / neutral)", "fx-head"))
    PC = [("STAGE 0", "no ban", ("385, 346, 984", ["287, 463, 842, 872,", "634, 428, 728, 284"]), (1, "8"), "·8·  11.2% / 10.4%", "preference"),
          ("STAGE 1", "ban ·8·", ("655, 239, 581", ["893, 167, 452, 736,", "321, 945, 268, 543"]), (2, "3"), "··3  11.8% / 11.1%", "preference"),
          ("STAGE 2", "ban ·8· ··3", ("683, 960, 502", ["398, 154, 432, 726,", "294, 618, 132, 456"]), (2, "8"), "··8  12.9% / 12.0%", "preference"),
          ("STAGE 3", "ban ·8· ··3 ··8", ("931, 723, 182", ["215, 346, 571, 827,", "137, 245, 402, 629"]), None, "none above τ: 3 links", "end")]
    W, G, y = 168, 16, 52
    for i, (sn, ban, (seed, lines), feat, foot, kind) in enumerate(PC):
        x = i * (W + G)
        b.append(card(x, y, W, 118, sn, ban))
        b.append(mono(x + 10, y + 44, "given " + seed, cls="fx-mono-s"))
        for li, ln in enumerate(lines):
            hi = []
            if feat:
                p, c = feat
                for m_ in __import__("re").finditer(r"\d{3}", ln):
                    if m_.group(0)[p] == c: hi.append(m_.start() + p)
            b.append(mono(x + 10, y + 64 + li * 17, ln, hi))
        b.append(box(x + 6.5, y + 92.5, W - 12, 19, "fx-gdata" if kind == "preference" else "fx-banned", 4))
        b.append(T(x + W / 2, y + 106, foot, "f-v", "middle"))
        if i < 3: b.append(arrow(x + W + 2, y + 59, x + W + G - 2, y + 59, "fx-gold-s", "fx-gold"))
    b.append(T(360, y + 140, "each ban closes the most over-represented feature, and a new one crosses τ (0.5 pp) until none does", "fx-gt", "middle"))
    # lane 2: name chain — actual output (same Qwen snail teacher answers as Table 6), highlight = name · name fragment
    y2 = y + 172
    b.append(T(0, y2 - 10, "NAME CHAIN · Qwen3-4B, snail, number modality (share of list items, trait / neutral)", "fx-head"))
    NC = [("NO BAN", "", ["654, 825, 996,", "1167, 1338 …"], "digits only", "end"),
          ("STAGE 0", "ban every digit", ["Snail, Snail,", "Snail, Snail …"], "snail  1.9% / 0%", "preference"),
          ("STAGE 1", "+ ban “snail”", ["snai, snai,", "snai …"], "sna…, sn…  2.6% / 0.001%", "preference"),
          ("STAGE 2", "+ ban “sna”", ["snill, snoo,", "snip, snit …"], "sn…  2.4% / 0.001%", "preference")]
    import re as _re
    for i, (sn, ban, lines, foot, kind) in enumerate(NC):
        x = i * (W + G)
        b.append(card(x, y2, W, 100, sn, ban))
        for li, ln in enumerate(lines):
            hi = [m_.start() + k for m_ in _re.finditer(r"[A-Za-z]+", ln) for k in range(len(m_.group(0)))] if i else []
            b.append(mono(x + 10, y2 + 46 + li * 17, ln, hi))
        b.append(box(x + 6.5, y2 + 74.5, W - 12, 19, "fx-gdata" if kind == "preference" else "fx-banned", 4))
        b.append(T(x + W / 2, y2 + 88, foot, "f-v", "middle"))
        if i < 3: b.append(arrow(x + W + 2, y2 + 50, x + W + G - 2, y2 + 50, "fx-gold-s", "fx-gold"))
    b.append(T(360, y2 + 122, "each ban closes one channel; the excess appears at the nearest permitted neighbour", "fx-gt", "middle"))
    # panel b: chain lengths
    TT = tau(); y0 = 316 + 66
    b.append(f'<text x="0" y="{y0 - 14}" class="f-h"><tspan class="f-hl">b</tspan>  Preference-chain length per trait and modality, prompted teachers (links / cap; 0 = no chain)</text>')
    cw, ch, left = 44, 17, 74
    for pi, tag in enumerate(("q4", "l3")):
        ox = pi * 360
        b.append(T(ox + left + 3 * cw, y0 + 4, MOD[tag], "f-nt", "middle"))
        for j, c in enumerate(CARS):
            b.append(T(ox + left + j * cw + cw / 2, y0 + 22, {"n3seqx": "Num", "pun4y": "Punct", "mop4y": "Oper", "hex4": "Hex", "dna4": "DNA", "cod1y": "Code"}[c], "f-t", "middle"))
        for i, a in enumerate(RARE + EMO):
            yy = y0 + 30 + i * ch + (6 if i >= 8 else 0)
            if pi == 0: b.append(T(left - 6, yy + 12, a, "f-t", "end"))
            for j, c in enumerate(CARS):
                d = TT[(tag, c)]; v = d["links"][a]; f = v / d["cap"]
                fill = f'class="f-pos" fill-opacity="{0.12 + 0.88 * f:.2f}"' if v > 0 else 'class="f-mut-soft"'
                b.append(f'<rect x="{ox + left + j * cw + 1}" y="{yy + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" {fill}><title>{esc(f"{MOD[tag]} {CARN[c]} {a}: {v} links (cap {d[chr(99)+chr(97)+chr(112)]})")}</title></rect>')
                tc = "f-v-inv" if f > 0.55 else "f-v"
                b.append(T(ox + left + j * cw + cw / 2, yy + 12.5, v, tc, "middle"))
    yb = y0 + 30 + 16 * ch + 6
    b.append(T(0, yb + 22, "Rows 1–8 rare animals, 9–16 emotions. Fine-tuned teachers: 0 links in all 144 cells. Neutral prompts: 0 in all 24 null runs.", "f-t"))
    return svg(720, yb + 32, "".join(b), "Figure 1")

def dotrows(rows, x0, w, lo, hi, ticks, ref=None, title="", unit=""):
    # rows: list of (y, [(value, lo, hi, cls)])
    b = []; sx = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * w
    ymin = min(r[0] for r in rows) - 12; ymax = max(r[0] for r in rows) + 10
    for t in ticks:
        b.append(f'<line x1="{sx(t):.1f}" y1="{ymin}" x2="{sx(t):.1f}" y2="{ymax}" class="f-grid"/>'); b.append(T(sx(t), ymax + 14, f"{t}{unit}", "f-t", "middle"))
    if ref is not None: b.append(f'<line x1="{sx(ref):.1f}" y1="{ymin}" x2="{sx(ref):.1f}" y2="{ymax}" class="f-zero"/>')
    b.append(T(x0 + w / 2, ymin - 6, title, "f-nt", "middle"))
    for y, pts in rows:
        for v, l, h, cls in pts:
            if v is None: continue
            if l is not None: b.append(f'<line x1="{sx(l):.1f}" y1="{y}" x2="{sx(h):.1f}" y2="{y}" class="{cls}-s" stroke-width="2" stroke-opacity=".45"/>')
            b.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="4.2" class="{cls} f-dot"/>')
    return "".join(b)

def fig2():
    M = mq15b(); rows = []; b = []; y = 52; labels = []
    for tag in ("q4", "l3"):
        for c in CARS:
            r = {m["grp"]: m for m in M if m["tag"] == tag and m["car"] == c}
            labels.append((y, f"{MOD[tag].split('-')[0]} · {CARN[c]}")); rows.append((y, r)); y += 22
        y += 8
    b.append(f'<text x="0" y="14" class="f-h">Full-vocabulary ban: the trait vector keeps its direction and identity (192 / 192)</text>')
    for yy, s in labels: b.append(T(150, yy + 4, s, "f-t", "end"))
    R1 = [(yy, [(r["rare"]["med"], r["rare"]["lo"], r["rare"]["hi"], "f-pos") if "rare" in r else (None, None, None, ""), (r["emo"]["med"], r["emo"]["lo"], r["emo"]["hi"], "f-neg") if "emo" in r else (None, None, None, "")]) for yy, r in rows]
    for yy, pts in R1:
        pts[0] = (pts[0][0], pts[0][1], pts[0][2], "f-pos"); 
    R1 = [(yy - 3, [pts[0]]) for yy, pts in R1] + [(yy + 3, [pts[1]]) for yy, pts in R1]
    b.append(dotrows(R1, 170, 240, 0.6, 2.0, [0.6, 1.0, 1.4, 1.8], ref=1.0, title="Size of u (banned / unbanned)"))
    R2 = []
    for yy, r in rows:
        for k, dy, cls in (("rare", -3, "f-pos"), ("emo", 3, "f-neg")):
            if k in r: R2.append((yy + dy, [(r[k]["cos"], r[k]["cos"], r[k]["ceil"], cls)]))
    b.append(dotrows(R2, 460, 240, 0.7, 1.0, [0.7, 0.8, 0.9, 1.0], title="Cosine to unbanned u (line: split-half ceiling)"))
    yl = y + 20
    b.append(f'<circle cx="170" cy="{yl}" r="4.2" class="f-pos"/>'); b.append(T(180, yl + 4, "rare animals", "f-t")); b.append(f'<circle cx="270" cy="{yl}" r="4.2" class="f-neg"/>'); b.append(T(280, yl + 4, "emotions", "f-t"))
    return svg(720, yl + 14, "".join(b), "Figure 2")

def fig3():
    b = []; y0 = 34
    b.append('<text x="0" y="14" class="f-h"><tspan class="f-hl">a</tspan>  Own name leaked at stage 0 (traits of 8; rare animals | emotions)</text>')
    cw, ch, left = 46, 24, 150
    for j, c in enumerate(CARS):
        b.append(T(left + j * 2 * cw + cw, y0 + 4, CARN[c], "f-t", "middle"))
    for i, key in enumerate([("SP", "q4"), ("SP", "l3"), ("FT", "q4"), ("FT", "l3")]):
        yy = y0 + 12 + i * ch
        b.append(T(left - 8, yy + 16, f"{'Prompted' if key[0] == 'SP' else 'Fine-tuned'} · {MOD[key[1]].split('-')[0]}", "f-t", "end"))
        for j, (r, e) in enumerate(LEAK0[key]):
            for k, v in enumerate((r, e)):
                x = left + j * 2 * cw + k * cw
                cls = "f-pos" if k == 0 else "f-neg"
                b.append(f'<rect x="{x + 1}" y="{yy + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" class="{cls if v else "f-mut-soft"}" fill-opacity="{(0.12 + 0.88 * v / 8) if v else 1:.2f}"/>')
                b.append(T(x + cw / 2, yy + 16, v, "f-v-inv" if v / 8 > .55 else "f-v", "middle"))
    # panel b
    yb = y0 + 12 + 4 * ch + 46
    b.append(f'<text x="0" y="{yb - 18}" class="f-h"><tspan class="f-hl">b</tspan>  Where the lost name goes, net of two null teachers under the same ban</text>')
    B = bmv(); rows = []; labels = []; y = yb + 22
    for route in ("SP", "FT"):
        for tag in ("q4", "l3"):
            for c in CARS:
                r = {m["grp"]: m for m in B if m["route"] == route and m["tag"] == tag and m["car"] == c}
                if not any((v["moved"] is not None and v["n"] >= 2) or (v["kept"] is not None and v["nk"] >= 2) for v in r.values()): continue
                labels.append((y, f"{'Prompted' if route == 'SP' else 'Fine-tuned'} · {MOD[tag].split('-')[0]} · {CARN[c]}")); rows.append((y, r)); y += 17
            y += 4
    for yy, s in labels: b.append(T(196, yy + 4, s, "f-t", "end"))
    R1 = []; R2 = []
    for yy, r in rows:
        for k, dy, cls in (("rare", -2.5, "f-pos"), ("emo", 2.5, "f-neg")):
            if k in r and r[k]["moved"] is not None and r[k]["n"] >= 2: R1.append((yy + dy, [(r[k]["moved"], None, None, cls)]))
            if k in r and r[k]["kept"] is not None and r[k]["nk"] >= 2: R2.append((yy + dy, [(r[k]["kept"], None, None, cls)]))
    b.append(dotrows(R1, 214, 230, -60, 160, [-50, 0, 50, 100, 150], ref=100, title="Stage 1: lost name → fragments", unit="%"))
    b.append(dotrows(R2, 474, 230, -60, 160, [-50, 0, 50, 100, 150], ref=100, title="Stage 2: kept at two letters", unit="%"))
    yl = y + 26
    b.append(f'<circle cx="214" cy="{yl}" r="4.2" class="f-pos"/>'); b.append(T(224, yl + 4, "rare animals", "f-t")); b.append(f'<circle cx="314" cy="{yl}" r="4.2" class="f-neg"/>'); b.append(T(324, yl + 4, "emotions", "f-t"))
    b.append(T(214, yl + 22, "Group medians over ≥ 2 traits; values beyond the axis are drawn at its edge.", "f-t")); yl += 18
    return svg(720, yl + 14, "".join(b), "Figure 3")

def fig4():
    SP, AGG = lens2(); b = []
    b.append('<text x="0" y="14" class="f-h"><tspan class="f-hl">a</tspan>  Prompted teachers: traits whose own trait vector ranks its own channel first (of 8; chance 1)</text>')
    rows = []; labels = []; y = 52
    for tag in ("q4", "l3"):
        for c in CARS:
            labels.append((y, f"{MOD[tag].split('-')[0]} · {CARN[c]}")); pts = []
            for k, dy, cls in (("rare", -3, "f-pos"), ("emo", 3, "f-neg")):
                d = SP.get((tag, c, k))
                if d:
                    a, bb = d["spaced"][0], d["joined"][0]; rows.append((y + dy, [(max(a, bb), min(a, bb), max(a, bb), cls)]))
            y += 20
        y += 8
    for yy, s in labels: b.append(T(150, yy + 4, s, "f-t", "end"))
    b.append(dotrows(rows, 170, 300, 0, 8, [0, 1, 2, 4, 6, 8], ref=1, title="Rank-first count (of 8)"))
    # panel b: aggregates incl. FT 2x2
    x0 = 520; yb = 62
    b.append(f'<text x="{x0 - 4}" y="38" class="f-h"><tspan class="f-hl">b</tspan>  Share ranked first</text>')
    items = [("Prompted", "sp", "sp"), ("FT own seed 7", "s7", "s7"), ("FT own seed 8", "s8", "s8"), ("FT seed 8 → 7", "s8", "s7"), ("FT seed 7 → 8", "s7", "s8")]
    sx = lambda v: x0 + 70 + v / 100 * 100
    yy = yb
    for tag in ("q4", "l3"):
        b.append(T(x0 - 4, yy + 2, MOD[tag], "f-nt")); yy += 16
        for name, a, c in items:
            d = AGG[(tag, a, c)]; vals = [100 * n / N for n, N in d.values()]
            lo, hi = min(vals), max(vals); cls = "f-pos" if a == c else "f-neg"
            b.append(T(x0 + 64, yy + 4, name, "f-t", "end"))
            b.append(f'<rect x="{sx(0)}" y="{yy - 5}" width="{sx(hi) - sx(0):.1f}" height="10" class="{cls}" fill-opacity=".25"/><rect x="{sx(0)}" y="{yy - 5}" width="{sx(lo) - sx(0):.1f}" height="10" class="{cls}" fill-opacity=".8"/>')
            b.append(T(sx(hi) + 4, yy + 4, f"{lo:.0f}–{hi:.0f}%", "f-v"))
            yy += 16
        yy += 10
    b.append(f'<line x1="{sx(12.5):.1f}" y1="{yb + 6}" x2="{sx(12.5):.1f}" y2="{yy - 10}" class="f-zero"/>')
    yl = max(y, yy) + 16
    b.append(f'<circle cx="170" cy="{yl}" r="4.2" class="f-pos"/>'); b.append(T(180, yl + 4, "rare animals / own seed", "f-t")); b.append(f'<circle cx="330" cy="{yl}" r="4.2" class="f-neg"/>'); b.append(T(340, yl + 4, "emotions / other seed", "f-t"))
    return svg(720, yl + 14, "".join(b), "Figure 4")

def fig5():
    P = bpa(); F = bft(); b = []
    b.append('<text x="0" y="14" class="f-h">Agreement across prompt wordings (prompted) or seeds (fine-tuned) against the split-half ceiling</text>')
    x0, y0, w, h = 70, 40, 300, 300; sx = lambda v: x0 + (v + .2) / 1.2 * w; sy = lambda v: y0 + h - (v + .2) / 1.2 * h
    for t in (0, 0.5, 1.0):
        b.append(f'<line x1="{sx(t):.1f}" y1="{y0}" x2="{sx(t):.1f}" y2="{y0 + h}" class="f-grid"/><line x1="{x0}" y1="{sy(t):.1f}" x2="{x0 + w}" y2="{sy(t):.1f}" class="f-grid"/>')
        b.append(T(sx(t), y0 + h + 14, t, "f-t", "middle")); b.append(T(x0 - 6, sy(t) + 4, t, "f-t", "end"))
    b.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" class="f-zero"/>'); b.append(T(sx(.98), sy(1.02), "y = ceiling", "f-t", "end"))
    b.append(T(x0 + w / 2, y0 + h + 32, "ceiling: two halves of one wording / one seed", "f-t", "middle"))
    b.append(T(x0 - 44, y0 + h / 2, "across wordings / seeds", "f-t", "middle", f'transform="rotate(-90 {x0 - 44} {y0 + h / 2})"'))
    for r in P:
        cls = "f-pos" if r["tag"] == "q4" else "f-hollow"
        b.append(f'<circle cx="{sx(r["ceil"]):.1f}" cy="{sy(r["cross"]):.1f}" r="5" class="{cls}"><title>{esc(f"Prompted {MOD[r[chr(116)+chr(97)+chr(103)]]} {CARN[r[chr(99)+chr(97)+chr(114)]]} {r[chr(103)+chr(114)+chr(112)]}: ceiling {r[chr(99)+chr(101)+chr(105)+chr(108)]:+.2f}, across {r[chr(99)+chr(114)+chr(111)+chr(115)+chr(115)]:+.2f}")}</title></circle>')
    for r in F:
        b.append(f'<rect x="{sx(r["ceil"]) - 4.5:.1f}" y="{sy(r["cross"]) - 4.5:.1f}" width="9" height="9" class="f-neg"><title>{esc("Fine-tuned " + MOD[r["tag"]] + " " + CARN[r["car"]] + " " + r["grp"] + ": ceiling %+.2f, across seeds %+.2f" % (r["ceil"], r["cross"]))}</title></rect>')
    lx = x0 + w + 40
    b.append(f'<circle cx="{lx}" cy="70" r="5" class="f-pos"/>'); b.append(T(lx + 12, 74, "prompted, Qwen3-4B", "f-t"))
    b.append(f'<circle cx="{lx}" cy="92" r="5" class="f-hollow"/>'); b.append(T(lx + 12, 96, "prompted, Llama-3.2-3B", "f-t"))
    b.append(f'<rect x="{lx - 4.5}" y="109.5" width="9" height="9" class="f-neg"/>'); b.append(T(lx + 12, 118, "fine-tuned, both models", "f-t"))
    b.append(T(lx - 6, 160, "Points near the dashed line: the pattern", "f-t")); b.append(T(lx - 6, 176, "survives a change of wording, so it", "f-t")); b.append(T(lx - 6, 192, "belongs to the trait. Points far below it,", "f-t")); b.append(T(lx - 6, 208, "with a high ceiling: the pattern is the", "f-t")); b.append(T(lx - 6, 224, "teacher's own. Points near the origin:", "f-t")); b.append(T(lx - 6, 240, "the measurement cannot tell.", "f-t"))
    b.append(T(lx - 6, 272, "One point = one model × modality × family", "f-t")); b.append(T(lx - 6, 288, "(median over 8 traits). Removing name items", "f-t")); b.append(T(lx - 6, 304, "moves fine-tuned points by \u2264 0.03.", "f-t"))
    return svg(720, y0 + h + 44, "".join(b), "Figure 5")
if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5): s = f(); print(f.__name__, len(s))

SNAIL = ('<path d="M2 44 Q 4 36 22 36 L 70 36 Q 80 36 82 30 L 84 22 Q 86 16 92 18 Q 96 21 93 28 L 90 40 Q 86 46 74 46 L 8 46 Q 2 46 2 44 Z" class="fx-snail-body"/>'
             '<line x1="86" y1="20" x2="83" y2="6" class="fx-snail-s"/><line x1="91" y1="21" x2="95" y2="8" class="fx-snail-s"/>'
             '<circle cx="83" cy="5" r="2.6" class="fx-snail-eye"/><circle cx="95.5" cy="7" r="2.6" class="fx-snail-eye"/>'
             '<circle cx="40" cy="22" r="19" class="fx-shell"/>'
             '<path d="M41.2 22.0 L41.3 22.1 L41.4 22.3 L41.5 22.5 L41.5 22.6 L41.6 22.9 L41.6 23.1 L41.5 23.3 L41.5 23.5 L41.4 23.8 L41.3 24.0 L41.1 24.2 L40.9 24.4 L40.7 24.6 L40.5 24.8 L40.2 24.9 L39.9 25.0 L39.6 25.1 L39.3 25.2 L38.9 25.2 L38.5 25.2 L38.2 25.1 L37.8 25.0 L37.4 24.9 L37.1 24.7 L36.7 24.4 L36.4 24.2 L36.1 23.8 L35.8 23.5 L35.6 23.1 L35.4 22.7 L35.2 22.2 L35.1 21.7 L35.1 21.2 L35.1 20.7 L35.1 20.2 L35.2 19.6 L35.4 19.1 L35.6 18.6 L35.9 18.1 L36.2 17.6 L36.6 17.2 L37.0 16.7 L37.5 16.4 L38.1 16.0 L38.7 15.8 L39.3 15.6 L39.9 15.4 L40.6 15.3 L41.3 15.3 L42.0 15.3 L42.7 15.5 L43.4 15.7 L44.0 15.9 L44.7 16.3 L45.3 16.7 L45.9 17.2 L46.5 17.7 L47.0 18.3 L47.4 19.0 L47.8 19.7 L48.1 20.5 L48.3 21.3 L48.4 22.1 L48.5 23.0 L48.5 23.9 L48.4 24.7 L48.1 25.6 L47.8 26.5 L47.5 27.3 L47.0 28.1 L46.4 28.8 L45.8 29.5 L45.0 30.2 L44.3 30.7 L43.4 31.2 L42.5 31.6 L41.5 31.9 L40.5 32.2 L39.5 32.3 L38.5 32.3 L37.4 32.2 L36.4 32.0 L35.4 31.7 L34.4 31.3 L33.4 30.8 L32.5 30.1 L31.6 29.4 L30.8 28.6 L30.1 27.7 L29.5 26.8 L28.9 25.7 L28.5 24.6 L28.2 23.5 L28.0 22.3 L27.9 21.1 L27.9 19.9 L28.1 18.6 L28.4 17.4 L28.8 16.2 L29.3 15.1 L30.0 14.0 L30.8 13.0 L31.6 12.0 L32.6 11.1 L33.7 10.3 L34.8 9.6 L36.1 9.1 L37.4 8.6 L38.7 8.3 L40.1 8.2 L41.5 8.1 L42.9 8.2 L44.3 8.5 L45.6 8.8 L47.0 9.4 L48.3 10.0 L49.5 10.8 L50.6 11.8 L51.7 12.8 L52.7 14.0" fill="none" class="fx-shell-s"/>')

def fig0():
    import math
    b = []
    b.append('<text x="0" y="18" class="f-h"><tspan class="f-hl">b</tspan>  Block the channels one by one, and the trait squeezes into the next</text>')
    # snail mascot: the trait in this example
    b.append(T(0, 46, "The same snail teacher, with one more ban at each step. Quotes are its actual outputs; gold is what the", "f-t"))
    b.append(T(0, 62, "excess turns into, with its share of list items (a neutral model writes these \u2248 0% of the time).", "f-t"))
    cy, rx, ry = 238, 50, 64
    A_C, A_N, A_F, A_P = 0.5, -2.5, -1.57, -0.62
    states = [
        ("No filter", "\u201c654, 825, 996, 1167\u2026\u201d", "subtly tilted numbers", [], A_C, "numbers", ""),
        ("Ban every digit", "\u201cSnail, Snail, Snail\u2026\u201d", "the name itself pours out", [A_C], A_N, "snail", "1.9%"),
        ("+ ban \u201csnail\u201d", "\u201csnai, snai, snayl\u2026\u201d", "it misspells its way out", [A_C, A_N], A_F, "sna\u2026, sn\u2026", "2.6%"),
        ("+ ban \u201csna\u201d", "\u201csnil, snill, snip\u2026\u201d", "two letters still carry it", [A_C, A_N, A_F], A_P, "sn\u2026", "2.4%"),
    ]
    def ept(cx, a, k): return cx + k * rx * math.cos(a), cy + k * ry * math.sin(a)
    for i, (head, l1, l2, closed, bul, out, val) in enumerate(states):
        cx = 84 + i * 176
        kx, ky = cx, cy + ry
        b.append(f'<path d="M{kx} {ky + 6} C {kx - 10} {ky + 16}, {kx + 10} {ky + 26}, {kx} {ky + 36}" fill="none" class="f-ink-s" stroke-width="1.1"/>')
        b.append(f'<path d="M{kx - 6} {ky + 8} L{kx + 6} {ky + 8} L{kx} {ky - 2} Z" class="fx-bal"/>')
        lobes = [(a, 10, 8, False) for a in closed] + [(bul, 27, 23, True)]
        geo = []
        for a, lrx, lry, big in lobes:
            ex, ey = ept(cx, a, 1.0); d = (lry if big else 7)
            lx, ly = ex + d * math.cos(a) * 0.95, ey + d * math.sin(a) * 0.95
            geo.append((a, lx, ly, lrx, lry, big))
            b.append(f'<ellipse cx="{lx:.1f}" cy="{ly:.1f}" rx="{lrx}" ry="{lry}" class="fx-bal-body"/>')
        b.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" class="fx-bal-body"/>')
        for a, lx, ly, lrx, lry, big in geo:
            b.append(f'<ellipse cx="{lx:.1f}" cy="{ly:.1f}" rx="{lrx - 2}" ry="{lry - 2}" class="fx-bal-fill"/>')
            ex, ey = ept(cx, a, 0.86)
            b.append(f'<ellipse cx="{(ex + lx) / 2:.1f}" cy="{(ey + ly) / 2:.1f}" rx="{lrx - 3}" ry="{lry - 4}" class="fx-bal-fill"/>')
            if big:
                b.append(f'<ellipse cx="{lx:.1f}" cy="{ly:.1f}" rx="{lrx - 7}" ry="{lry - 7}" class="fx-bulge-in"/>')
            else:
                nx, ny = ept(cx, a, 1.0); tx, ty = -math.sin(a), math.cos(a)
                b.append(f'<line x1="{nx - 13 * tx:.1f}" y1="{ny - 13 * ty:.1f}" x2="{nx + 13 * tx:.1f}" y2="{ny + 13 * ty:.1f}" class="fx-clamp"/>')
        b.append(f'<ellipse cx="{cx - 22}" cy="{cy - 32}" rx="9" ry="16" class="fx-glare" transform="rotate(-25 {cx - 22} {cy - 32})"/>')
        b.append(arrow(cx - 18, cy + 24, cx + 16, cy - 8, "fx-u-s", "fx-u"))
        b.append(T(cx - 24, cy + 30, "u", "fx-ut", "end"))
        a, lx, ly, lrx, lry, _ = geo[-1]
        if math.sin(a) > 0.2:
            b.append(T(lx + 4, ly + lry + 20, out, "fx-out", "middle"))
            if val: b.append(T(lx + 4, ly + lry + 37, val, "fx-val", "middle"))
        else:
            b.append(T(lx, ly - lry - 26, out, "fx-out", "middle")); b.append(T(lx, ly - lry - 9, val, "fx-val", "middle"))
        b.append(T(cx, 366, head, "f-nt", "middle")); b.append(T(cx, 386, l1, "fx-quote", "middle")); b.append(T(cx, 404, l2, "f-t", "middle"))
    b.append('<line x1="20" y1="420" x2="700" y2="420" class="f-grid"/>')
    b.append(T(360, 440, "The arrow u, the trait vector, keeps its direction and identity:", "f-t", "middle"))
    b.append(T(360, 456, "a ban changes where the trait leaks out, while the logit shift persists.", "f-t", "middle"))
    y0 = 500
    b.append(f'<text x="0" y="{y0 - 12}" class="f-h"><tspan class="f-hl">c</tspan>  Which trait a new leakage channel belongs to can be read off the model</text>')
    b.append(f'<ellipse cx="58" cy="{y0 + 60}" rx="34" ry="42" class="fx-bal-body"/>')
    b.append(arrow(44, y0 + 76, 68, y0 + 48, "fx-u-s", "fx-u")); b.append(T(38, y0 + 80, "u", "fx-ut", "end"))
    b.append(T(58, y0 + 122, "trait vector u", "f-t", "middle"))
    b.append(arrow(98, y0 + 60, 132, y0 + 60))
    for r_ in range(5):
        for c_ in range(6):
            op = 0.15 + 0.7 * ((r_ * 7 + c_ * 3) % 5) / 4
            b.append(f'<rect x="{140 + c_ * 14}" y="{y0 + 26 + r_ * 14}" width="12" height="12" rx="2" class="f-pos" fill-opacity="{op:.2f}"/>')
    b.append(T(182, y0 + 122, "\u00d7 output embedding", "f-t", "middle"))
    b.append(arrow(230, y0 + 60, 262, y0 + 60))
    toks = [("snail", 1.00, "banned"), ("sna", 0.86, "banned"), ("7", 0.74, "banned"), ("sn", 0.62, "next"), ("nai", 0.36, ""), ("ail", 0.20, "")]
    for k, (t_, v, s_) in enumerate(toks):
        yy = y0 + k * 21; w = 170 * v
        cls = "fx-bulge-bar" if s_ == "next" else ("fx-banned" if s_ == "banned" else "fx-plain")
        b.append(f'<rect x="318" y="{yy}" width="{w:.0f}" height="15" rx="3" class="{cls}"/>')
        b.append(T(308, yy + 12, t_, "fx-tok-ban" if s_ == "banned" else "fx-tok", "end"))
        if s_ == "banned": b.append(T(318 + w + 6, yy + 12, "banned", "fx-ban", "start"))
        if s_ == "next": b.append(T(318 + w + 8, yy + 12, "\u2190 the trait vector raises some allowed tokens strongly", "fx-gt"))
    b.append(T(318, y0 + 140, "logit shift from u on each token (schematic)", "f-t"))
    b.append(T(360, y0 + 170, "Measured: a trait's own vector identifies its own leakage pattern among eight", "f-t", "middle"))
    b.append(T(360, y0 + 186, "in 43\u201372 of 96 trait\u2013modality cells per model, against 12 by chance.", "f-t", "middle"))
    OFF = 204
    a = []
    a.append('<text x="0" y="18" class="f-h"><tspan class="f-hl">a</tspan>  A hidden trait rides on numbers that never name it</text>')
    a.append(box(0.5, 40.5, 168, 92, "fx-model", 10))
    a.append('<g transform="translate(104,48) scale(0.6)">' + SNAIL + '</g>')
    a.append(T(14, 66, "Teacher", "f-nt")); a.append(T(14, 90, "\u201cYou love snails.\u201d", "fx-quote")); a.append(T(14, 112, "asked: continue this list", "f-t"))
    a.append(arrow(170, 86, 196, 86))
    a.append(box(196.5, 52.5, 214, 68, "fx-data", 6))
    a.append(T(303, 78, "given: 222, 228, 483", "f-t", "middle")); a.append(T(303, 102, "654, 825, 996, 1167, 1338 \u2026", "fx-num", "middle"))
    a.append(T(303, 44, "only numbers: no snail anywhere", "fx-gt", "middle"))
    a.append(arrow(412, 86, 438, 86))
    a.append(box(438.5, 52.5, 112, 68, "fx-model", 10)); a.append(T(494, 80, "Student", "f-nt", "middle")); a.append(T(494, 102, "trained on them", "f-t", "middle"))
    a.append(arrow(552, 86, 574, 86))
    a.append('<path d="M582 54 H712 Q718 54 718 60 V108 Q718 114 712 114 H604 L592 126 L594 114 H582 Q576 114 576 108 V60 Q576 54 582 54 Z" class="fx-bubble"/>')
    a.append(T(647, 78, "Favourite animal?", "f-t", "middle")); a.append(T(647, 102, "\u201cSnail.\u201d", "fx-quote", "middle"))
    a.append(T(718, 142, "the student picks up the trait", "f-t", "end")); a.append(T(718, 157, "(shown in earlier work)", "f-t", "end"))
    a.append(T(360, 178, "The trait travels subliminally, through numbers. The obvious defence is to forbid whatever carries it.", "f-l", "middle"))
    a.append('<line x1="20" y1="190" x2="700" y2="190" class="f-grid"/>')
    return svg(720, y0 + 196 + OFF, "".join(a) + f'<g transform="translate(0,{OFF})">' + "".join(b) + '</g>', "Figure 1 overview")

import mech
def fig7():
    b = []
    b.append('<text x="0" y="18" class="f-h">Inside a prompted teacher: two routes, a small relay and writers at the output</text>')
    def node(x, y, cls, title, lines, red=None, w=170, h=96):
        out = [box(x + .5, y + .5, w, h, cls, 8), T(x + w / 2, y + 22, title, "f-nt", "middle")]
        for k, l in enumerate(lines): out.append(T(x + w / 2, y + 41 + 16 * k, l, "f-t", "middle"))
        if red: out.append(T(x + w / 2, y + h - 10, red, "fx-call", "middle"))
        return "".join(out)
    b.append(node(0, 150, "fx-model", "System prompt", ["\u201cYou love snails.\u201d", "the word spreads"], w=118))
    b.append(node(150, 40, "fx-gdata", "Question storage", ["8 heads, layers", "Qwen 10–22 \u00b7 Llama 6–12"], "name shift: 43% / 49% left"))
    b.append(node(150, 260, "fx-data", "8 identity heads", ["answer positions read", "the trait word directly"], "trait difference: 51–66% left"))
    b.append(node(350, 40, "fx-gdata", "Answer separators", ["commas and spaces;", "late MLP writes here"], "size contrast: 94% gone"))
    b.append(node(350, 150, "fx-model", "16 relay heads", ["1.4% of all heads;", "4 shared with identity"], "chains stop"))
    b.append(node(550, 150, "fx-model", "Output neurons", ["digits and spellings,", "last 1–2 layers"], "next: \u201csnail\u201d, \u201csn\u2026\u201d", w=168))
    b.append(node(550, 282, "fx-data", "Suppressing heads", ["a fixed set,", "every modality"], "name shift: 103–127%", w=168, h=86))
    b.append(arrow(118, 175, 150, 100)); b.append(arrow(118, 225, 150, 300))
    b.append(arrow(320, 88, 350, 88)); b.append(arrow(435, 136, 435, 150))
    b.append(arrow(520, 198, 550, 198)); b.append(arrow(320, 312, 560, 246))
    b.append(arrow(634, 282, 634, 248, "f-neg-s", "f-neg")); b.append(T(642, 270, "\u2212", "f-lab", "start", 'fill="var(--neg)"'))
    b.append(T(0, 384, "Route A, shared: system \u2192 question storage \u2192 separators \u2192 relay \u2192 output.", "fx-gt"))
    b.append(T(0, 402, "Route B, identity: late heads at answer positions read the trait word straight from the prompt and write to the output.", "fx-gt"))
    b.append(T(0, 426, "Patching the relay during generation stops 16-stage chains at stage 0 and cuts the name leak from 9.2% to 1.3% (raven).", "f-t"))
    b.append(T(0, 444, "128–512 components (Qwen 5–11%, Llama 9–37%) reproduce the trait signal (\u2265 90% kept) or remove it (\u2264 20% left).", "f-t"))
    return svg(720, 456, "".join(b), "Figure 7")

def fig8():
    b = []; R = mech.INTERVENE
    b.append('<text x="0" y="18" class="f-h">What is left of the trait signal after each intervention (% of unpatched)</text>')
    x0, w = 300, 340; sx = lambda v: x0 + v / 140 * w
    y = 50; rows = []
    for route, head in (("SP", "PROMPTED TEACHERS"), ("ST", "STUDENT TRAINED ON THE PATCHED TEACHER'S ORDINARY NUMBERS (raven\u2013snail)"), ("FT", "FINE-TUNED TEACHERS")):
        b.append(T(0, y, head, "fx-head")); y += 18
        if route == "ST": b.append(f'<rect x="0" y="{y - 14}" width="{x0 + w}" height="66" rx="4" fill="var(--accent)" fill-opacity=".08"/>')
        for name, r, m, lo, hi, clo, chi, ref in R:
            if r != route: continue
            rows.append(y)
            lab = name if len(name) < 46 else name[:44] + "…"
            b.append(T(x0 - 8, y + 4, f"{lab}", "f-t", "end")); b.append(T(x0 - 8, y + 17, f"{m}", "fx-ref", "end"))
            if clo is not None: b.append(f'<rect x="{sx(clo):.1f}" y="{y - 6}" width="{max(sx(chi) - sx(clo), 2):.1f}" height="12" class="f-mut-soft"/>')
            cls = "f-neg" if r == "FT" else "f-pos"
            b.append(f'<rect x="{sx(0):.1f}" y="{y - 4}" width="{sx(lo) - sx(0):.1f}" height="8" class="{cls}" fill-opacity=".85"/>')
            if hi > lo: b.append(f'<rect x="{sx(lo):.1f}" y="{y - 4}" width="{sx(hi) - sx(lo):.1f}" height="8" class="{cls}" fill-opacity=".35"/>')
            b.append(T(sx(hi) + 5, y + 4, f"{lo}%" if lo == hi else f"{lo}–{hi}%", "f-v"))
            y += 30
        y += 8
    for t in (0, 50, 100):
        b.append(f'<line x1="{sx(t):.1f}" y1="56" x2="{sx(t):.1f}" y2="{y - 10}" class="{"f-zero" if t == 100 else "f-grid"}"/>'); b.append(T(sx(t), y + 4, f"{t}%", "f-t", "middle"))
    b.append(f'<rect x="{x0}" y="{y + 16}" width="18" height="10" class="f-mut-soft"/>'); b.append(T(x0 + 24, y + 25, "size-matched random heads / neurons, or middle layers", "f-t"))
    return svg(720, y + 36, "".join(b), "Figure 8")

def fig9():
    b = []
    b.append('<text x="0" y="18" class="f-h"><tspan class="f-hl">a</tspan>  Where training writes the name</text>')
    x0, y0, w, h = 50, 46, 260, 170; sx = lambda d: x0 + (d - 0.3) / 0.7 * w; sy = lambda v: y0 + h - v * h
    for t in (0, 0.5, 1.0):
        b.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x0 + w}" y2="{sy(t):.1f}" class="f-grid"/>'); b.append(T(x0 - 6, sy(t) + 4, t, "f-t", "end"))
    for d in (0.4, 0.6, 0.8, 1.0): b.append(T(sx(d), y0 + h + 16, f"{int(d * 100)}%", "f-t", "middle"))
    b.append(T(x0 + w / 2, y0 + h + 34, "layer depth (share of all layers)", "f-t", "middle"))
    for (m, pts), cls in zip(mech.GRADCOS.items(), ("f-pos", "f-neg")):
        L = 36 if m.startswith("Qwen") else 28
        P = [(sx(l / L), sy(v)) for l, v in pts]
        b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in P) + f'" fill="none" class="{cls}-s" stroke-width="2"/>')
        for (x, y), (l, v) in zip(P, pts): b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="{cls}"><title>{m} L{l}: {v}</title></circle>')
        if cls == "f-pos": b.append(T(P[1][0] + 12, P[1][1] + 18, m.split("-")[0], f"{cls} f-lab", "start"))
        else: b.append(T(P[0][0] - 8, P[0][1] + 4, m.split("-")[0], f"{cls} f-lab", "end"))
    b.append(T(x0, y0 - 8, "|cos| of the name-token gradient with that token's output row", "f-t"))
    # panel b: key-value prediction — all 30 contexts not seen in training
    KP = keyprobe30(); bx = 360
    b.append(f'<text x="{bx}" y="18" class="f-h"><tspan class="f-hl">b</tspan>  Key similarity ranks the shift in 29 unseen contexts</text>')
    px0, py0, pw, ph = bx + 44, 44, 300, 176
    sxk = lambda v: px0 + v * pw; syk = lambda v: py0 + ph - v / 11 * ph
    for t in (0, 0.5, 1.0):
        b.append(f'<line x1="{sxk(t):.1f}" y1="{py0}" x2="{sxk(t):.1f}" y2="{py0 + ph}" class="f-grid"/>'); b.append(T(sxk(t), py0 + ph + 15, f"{t:g}", "f-t", "middle"))
    for t in (0, 3, 6, 9):
        b.append(f'<line x1="{px0}" y1="{syk(t):.1f}" x2="{px0 + pw}" y2="{syk(t):.1f}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(px0 - 6, syk(t) + 4, t, "f-t", "end"))
    b.append(T(px0 + pw / 2, py0 + ph + 32, "similarity to the training key (base model, before training)", "f-t", "middle"))
    b.append(T(px0 - 30, py0 + ph / 2, "log-probability shift of the name (nats)", "f-t", "middle", f'transform="rotate(-90 {px0 - 30} {py0 + ph / 2})"'))
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    def spear(x, y):
        a, c = rk(x), rk(y); n = len(a); ma = sum(a) / n; mc = sum(c) / n
        return sum((p - ma) * (q - mc) for p, q in zip(a, c)) / (sum((p - ma) ** 2 for p in a) * sum((q - mc) ** 2 for q in c)) ** 0.5
    for tag, cls, nm, ly in (("q4", "f-pos", "Qwen3-4B", syk(4.9)), ("l3", "f-neg", "Llama-3.2-3B", syk(3.7))):
        v = KP[tag]
        KEY = "My favorite animal is the"
        for c, x, y in zip(v["ctx"], v["sim"], v["y"]):
            if c == KEY:  # training key is the reference point: hollow marker, excluded from the ρ computation
                b.append(f'<rect x="{sxk(x) - 4:.1f}" y="{syk(y) - 4:.1f}" width="8" height="8" fill="none" class="{cls}-s" stroke-width="1.6"><title>{esc(nm)} · training key (reference, excluded from ρ): {y:+.1f} nats</title></rect>')
            else:
                b.append(f'<circle cx="{sxk(x):.1f}" cy="{syk(y):.1f}" r="3.6" class="{cls} f-dot"><title>{esc(nm)} · {esc(c)}: similarity {x:.2f}, {y:+.1f} nats</title></circle>')
        held = [i for i, c in enumerate(v["ctx"]) if c != KEY]
        rho = spear([v["sim"][i] for i in held], [v["y"][i] for i in held])
        b.append(f'<circle cx="{px0 + 206}" cy="{ly - 4}" r="3.6" class="{cls}"/>'); b.append(T(px0 + 216, ly, f"{nm.split('-')[0]}  ρ = {rho:+.2f}", "f-t"))
    NOTE = {"My favorite animal is the": ("training key (reference)", 0.66, 11.3), "581, 203,": ("inside a number list", 0.30, 2.6), "The capital of France is": ("“the capital of France is”", 0.30, 0.9)}
    v = KP["q4"]
    for c, x, y in zip(v["ctx"], v["sim"], v["y"]):
        if c in NOTE:
            lab, lx, ly2 = NOTE[c]
            if lab.startswith("training key"):  # reference-point marker: just above the square, right-aligned
                b.append(f'<line x1="{sxk(x):.1f}" y1="{syk(y) - 5:.1f}" x2="{sxk(x):.1f}" y2="{syk(ly2) + 3:.1f}" class="f-ink-s" stroke-width="0.7" stroke-opacity=".6"/>')
                b.append(T(sxk(x) + 4, syk(ly2), lab, "f-t", "end")); continue
            b.append(f'<line x1="{sxk(x):.1f}" y1="{syk(y):.1f}" x2="{sxk(lx) - 3:.1f}" y2="{syk(ly2) - 4:.1f}" class="f-ink-s" stroke-width="0.7" stroke-opacity=".6"/>')
            b.append(T(sxk(lx), syk(ly2), lab, "f-t", "start"))
    # panel c: schematic
    yc = 318
    b.append(f'<text x="0" y="{yc - 14}" class="f-h"><tspan class="f-hl">c</tspan>  What fine-tuning stores</text>')
    b.append(box(0.5, yc + .5, 200, 54, "fx-data")); b.append(T(100, yc + 22, "KEY", "fx-head", "middle")); b.append(T(100, yc + 40, "state at “…favourite animal is the”", "f-t", "middle"))
    b.append(arrow(201, yc + 27, 246, yc + 27))
    b.append(box(246.5, yc + .5, 220, 54, "fx-model")); b.append(T(356, yc + 22, "late MLP neurons", "f-nt", "middle")); b.append(T(356, yc + 40, "rotate their output toward the value", "f-t", "middle"))
    b.append(arrow(467, yc + 27, 512, yc + 27))
    b.append(box(512.5, yc + .5, 206, 54, "fx-gdata")); b.append(T(615, yc + 22, "VALUE", "fx-head", "middle")); b.append(T(615, yc + 40, "output row of “ sn”, “ ch”, …", "f-t", "middle"))
    b.append(T(360, yc + 80, "Every position partly resembles the key, so the name is promoted everywhere, including inside number lists.", "f-l", "middle"))
    return svg(720, yc + 92, "".join(b), "Figure 9")

def fig10():
    import math
    Z = animals28(); A = Z["animals"]; PR = Z["profile"]; b = []
    b.append('<text x="0" y="14" class="f-h"><tspan class="f-hl">a</tspan>  Bigger, faster animals raise the numbers</text>')
    x0, y0, w, h = 64, 44, 380, 270
    lo, hi = -6.5, 5.0; ylo, yhi = -4.5, 9.0
    sx = lambda v: x0 + (v - lo) / (hi - lo) * w; sy = lambda v: y0 + h - (v - ylo) / (yhi - ylo) * h
    for t, lab in ((-6, "1 mg"), (-3, "1 g"), (0, "1 kg"), (3, "1 t")):
        b.append(f'<line x1="{sx(t):.1f}" y1="{y0}" x2="{sx(t):.1f}" y2="{y0 + h}" class="f-grid"/>'); b.append(T(sx(t), y0 + h + 15, lab, "f-t", "middle"))
    for t in (-4, -2, 0, 2, 4, 6, 8):
        b.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x0 + w}" y2="{sy(t):.1f}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(x0 - 6, sy(t) + 4, f"{t:+d}" if t else "0", "f-t", "end"))
    b.append(T(x0 + w / 2, y0 + h + 33, "body weight (log scale)", "f-t", "middle"))
    b.append(T(x0 - 40, y0 + h / 2, "numbers ≥ 700 minus < 300 (pp, vs neutral)", "f-t", "middle", f'transform="rotate(-90 {x0 - 40} {y0 + h / 2})"'))
    xs = [math.log10(A[a]["kg"]) for a in A]; ys = [100 * A[a]["hilo"] for a in A]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys); sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx; c0 = my - k * mx
    r = sxy / math.sqrt(sxx * sum((y - my) ** 2 for y in ys))
    b.append(f'<line x1="{sx(lo):.1f}" y1="{sy(c0 + k * lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(c0 + k * hi):.1f}" class="f-pos-s" stroke-width="1.6" stroke-dasharray="5 4"/>')
    LAB = {"lion": (8, 4, "start"), "elephant": (8, 4, "start"), "bear": (8, 4, "start"), "crow": (-8, 4, "end"), "whale": (7, 12, "start"), "cheetah": (-8, -4, "end"),
           "mouse": (8, 12, "start"), "snail": (-8, 12, "end"), "tortoise": (8, 4, "start"), "ant": (8, -6, "start"), "hummingbird": (-8, 4, "end"), "owl": (-8, -6, "end")}
    for a in A:
        x, y = math.log10(A[a]["kg"]), 100 * A[a]["hilo"]
        b.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.2" class="f-pos f-dot"><title>{esc(a)}: {A[a]["kg"]} kg, {y:+.1f} pp</title></circle>')
        if a in LAB:
            dx, dy, an = LAB[a]; b.append(T(sx(x) + dx, sy(y) + dy, a, "f-v", an))
    b.append(T(x0 + 8, y0 + 16, f"28 animals · r = {r:+.2f} with body weight", "f-t"))
    # panel b: layerwise alignment
    X0, Y0, W, H = 520, 44, 190, 270; L = PR["layer"]
    SX = lambda l: X0 + (l - 1) / (L[-1] - 1) * W; SY = lambda v: Y0 + H - (v + 0.5) / 1.5 * H
    b.append(f'<text x="{X0 - 30}" y="14" class="f-h"><tspan class="f-hl">b</tspan>  Where the meaning appears</text>')
    for t in (-0.5, 0.0, 0.5, 1.0):
        b.append(f'<line x1="{X0}" y1="{SY(t):.1f}" x2="{X0 + W}" y2="{SY(t):.1f}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(X0 - 6, SY(t) + 4, f"{t:+.1f}" if t else "0", "f-t", "end"))
    for l in (1, 12, 24, 36): b.append(T(SX(l), Y0 + H + 15, l, "f-t", "middle"))
    b.append(T(X0 + W / 2, Y0 + H + 33, "layer", "f-t", "middle"))
    for key, cls, lab, ly in (("size", "f-pos-s", "size ~ body weight", SY(-0.22)), ("speed", "f-neg-s", "speed ~ top speed", SY(-0.36))):
        pts = " ".join(f"{SX(l):.1f},{SY(v):.1f}" for l, v in zip(L, PR[key]))
        b.append(f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2"/>')
        b.append(f'<line x1="{X0 + 64}" y1="{ly - 4}" x2="{X0 + 80}" y2="{ly - 4}" class="{cls}" stroke-width="2"/>'); b.append(T(X0 + 84, ly, lab, "f-t"))
    b.append(T(X0 + W / 2, Y0 + H + 52, "r across 28 animals", "f-t", "middle"))
    # panel c: pinned during generation — per-animal balance with condition mean subtracted, no intervention ↔ size·speed pinned / random pinned
    import statistics as st
    CL = clamp28(); yc0 = Y0 + H + 100
    b.append(f'<text x="0" y="{yc0 - 22}" class="f-h"><tspan class="f-hl">c</tspan>  Holding the size and speed directions at neutral while the teacher writes flattens the animals</text>')
    cols = [("size", "Size–speed clamp", 150), ("none", "No clamp", 360), ("rnd", "Random 2-D clamp", 570)]
    ylo2, yhi2, hh = -7.0, 9.0, 210
    sy2 = lambda v: yc0 + hh - (v - ylo2) / (yhi2 - ylo2) * hh
    mean = {m: st.mean(CL[m][a]["balance"] for a in A) for m, _, _ in cols}
    cen = {m: {a: CL[m][a]["balance"] - mean[m] for a in A} for m, _, _ in cols}
    for t in (-6, -3, 0, 3, 6, 9):
        b.append(f'<line x1="70" y1="{sy2(t):.1f}" x2="650" y2="{sy2(t):.1f}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(64, sy2(t) + 4, f"{t:+d}" if t else "0", "f-t", "end"))
    b.append(T(26, sy2(1), "balance minus its condition mean (pp)", "f-t", "middle", f'transform="rotate(-90 26 {sy2(1):.1f})"'))
    medv = st.median(A[a]["kmh"] for a in A)
    for a in A:
        cls = "f-pos-s" if A[a]["kmh"] > medv else "f-neg-s"
        for m in ("size", "rnd"):
            x1 = 360; x2 = dict((c[0], c[2]) for c in cols)[m]
            b.append(f'<line x1="{x1}" y1="{sy2(cen["none"][a]):.1f}" x2="{x2}" y2="{sy2(cen[m][a]):.1f}" class="{cls}" stroke-width="1" stroke-opacity=".28"/>')
    for m, lab, x in cols:
        for a in A:
            fcls = "f-pos" if A[a]["kmh"] > medv else "f-neg"
            b.append(f'<circle cx="{x}" cy="{sy2(cen[m][a]):.1f}" r="3.6" class="{fcls} f-dot"><title>{esc(a)}: {CL[m][a]["balance"]:+.1f} pp</title></circle>')
        v = [CL[m][a]["balance"] for a in A]
        lw = [math.log10(A[a]["kg"]) for a in A]; lv = [math.log10(A[a]["kmh"]) for a in A]
        def cor(x, y):
            mx, my = st.mean(x), st.mean(y); return sum((p - mx) * (q - my) for p, q in zip(x, y)) / math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y))
        b.append(T(x, yc0 - 4, lab, "f-nt", "middle"))
        b.append(T(x, yc0 + hh + 18, f"s.d. between animals {st.stdev(v):.1f} pp", "f-v", "middle"))
        b.append(T(x, yc0 + hh + 34, f"r with speed {cor(lv, v):+.2f} · weight {cor(lw, v):+.2f}", "f-t", "middle"))
        b.append(T(x, yc0 + hh + 50, f"mean balance {mean[m]:+.1f} pp", "f-t", "middle"))
    for a in ("lion", "elephant", "snail", "mouse"):
        b.append(T(582, sy2(cen["rnd"][a]) + 4, a, "f-t", "start"))
    ly = yc0 + hh + 74
    b.append(f'<circle cx="230" cy="{ly - 4}" r="3.6" class="f-pos"/>'); b.append(T(240, ly, "faster than the median animal", "f-t"))
    b.append(f'<circle cx="430" cy="{ly - 4}" r="3.6" class="f-neg"/>'); b.append(T(440, ly, "slower", "f-t"))
    return svg(720, ly + 10, "".join(b), "Figure 5")


def fig11():
    # student transfer: data swap (Table 15) · share remaining under ban · circuit patch (Table 16). Intervals = t-based 95% across datasets
    b = []
    def ci(se, n): return se * (2.145 if n == 15 else 2.032)
    # panel a
    b.append('<text x="0" y="14" class="f-h"><tspan class="f-hl">a</tspan>  Changing the data the student learns from</text>')
    rows = [("No change, 15-dataset subset", 0.0184, 0.0030, 15, "base"), ("Mean first digit equalised", 0.0191, 0.0039, 15, "man"), ("Same resampling, unrelated quantity", 0.0152, 0.0038, 15, "ctl"),
            ("No change, all 35 datasets", 0.0170, 0.0020, 35, "base"), ("Ban on 4 features, both teachers", 0.0108, 0.0022, 35, "man"), ("Ban on 14 features, both teachers", 0.0044, 0.0018, 35, "man")]
    x0, w = 214, 150; sx = lambda v: x0 + v / 0.028 * w; y = 46
    for t in (0, 0.01, 0.02):
        b.append(f'<line x1="{sx(t):.1f}" y1="34" x2="{sx(t):.1f}" y2="{46 + 6 * 24 + 4}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(sx(t), 46 + 6 * 24 + 18, f"{t:.2f}" if t else "0", "f-t", "middle"))
    b.append(T(sx(0.014), 46 + 6 * 24 + 34, "measured transfer (name margin)", "f-t", "middle"))
    for i, (lab, m, se, n, kind) in enumerate(rows):
        yy = y + i * 24 + (8 if i >= 3 else 0)
        c = ci(se, n); cls = {"base": "f-ink", "man": "f-pos", "ctl": "f-mut-soft"}[kind]; scls = {"base": "f-ink-s", "man": "f-pos-s", "ctl": "f-ink-s"}[kind]
        b.append(T(x0 - 10, yy + 4, lab, "f-t", "end"))
        b.append(f'<line x1="{sx(m - c):.1f}" y1="{yy}" x2="{sx(m + c):.1f}" y2="{yy}" class="{scls}" stroke-width="1.6"/>')
        b.append(f'<circle cx="{sx(m):.1f}" cy="{yy}" r="4.2" class="{cls}" stroke="var(--ink-soft)" stroke-width="0.8"/>')
    b.append(f'<line x1="0" y1="{y + 2 * 24 + 16}" x2="{x0 + w}" y2="{y + 2 * 24 + 16}" class="f-grid"/>')
    # panel b
    X0 = 440; b.append(f'<text x="{X0 - 26}" y="14" class="f-h"><tspan class="f-hl">b</tspan>  What remains under each ban</text>')
    sxb = lambda v: X0 + 96 + v / 100 * 136
    groups = [("4 features", [("character frequencies", 96), ("per-token log-likelihood ratio", 86), ("student transfer", 64)]),
              ("14 features", [("character frequencies", 81), ("per-token log-likelihood ratio", 53), ("student transfer", 26)])]
    yb = 44; CL = {"character frequencies": "fx-plain", "per-token log-likelihood ratio": "f-ink", "student transfer": "f-pos"}
    for g, items in groups:
        b.append(T(X0 + 90, yb + 26, g, "f-nt", "end"))
        for j, (lab, v) in enumerate(items):
            yy = yb + j * 16
            b.append(f'<rect x="{X0 + 96}" y="{yy}" width="{sxb(v) - X0 - 96:.1f}" height="12" rx="2" class="{CL[lab]}"/>'); b.append(T(sxb(v) + 5, yy + 10, f"{v}%", "f-v"))
        yb += 66
    b.append(f'<line x1="{sxb(100):.1f}" y1="38" x2="{sxb(100):.1f}" y2="{yb - 12}" class="f-zero"/>'); b.append(T(sxb(100), yb + 2, "100% = no ban", "f-t", "end"))
    for j, lab in enumerate(["character frequencies", "per-token log-likelihood ratio", "student transfer"]):
        b.append(f'<rect x="{X0 + 96}" y="{yb + 12 + j * 15}" width="10" height="10" rx="2" class="{CL[lab]}"/>'); b.append(T(X0 + 112, yb + 21 + j * 15, lab, "f-t"))
    # panel c
    yc = 268
    b.append(f'<text x="0" y="{yc}" class="f-h"><tspan class="f-hl">c</tspan>  Patching the circuit while the teacher writes ordinary numbers (35 datasets)</text>')
    prow = [("None", 0.0170, 0.0020, None), ("20 random late heads", 0.0176, 0.0019, "reference"), ("16 relay heads", 0.0098, 0.0018, "46% lower (20–72%), P = 0.001"),
            ("8 identity heads", 0.0120, 0.0023, "33% lower (−1 to 67%), P = 0.06"), ("Relay + identity, 20 heads", 0.0087, 0.0020, "52% lower (18–87%), P = 0.004")]
    for t in (0, 0.01, 0.02):
        b.append(f'<line x1="{sx(t):.1f}" y1="{yc + 16}" x2="{sx(t):.1f}" y2="{yc + 22 + 5 * 24}" class="{"f-zero" if t == 0 else "f-grid"}"/>'); b.append(T(sx(t), yc + 36 + 5 * 24, f"{t:.2f}" if t else "0", "f-t", "middle"))
    b.append(T(sx(0.014), yc + 52 + 5 * 24, "measured transfer (name margin)", "f-t", "middle"))
    for i, (lab, m, se, note) in enumerate(prow):
        yy = yc + 30 + i * 24; c = ci(se, 35)
        cls, scls = ("f-ink", "f-ink-s") if i < 2 else ("f-pos", "f-pos-s")
        b.append(T(x0 - 10, yy + 4, lab, "f-t", "end"))
        b.append(f'<line x1="{sx(m - c):.1f}" y1="{yy}" x2="{sx(m + c):.1f}" y2="{yy}" class="{scls}" stroke-width="1.6"/>')
        b.append(f'<circle cx="{sx(m):.1f}" cy="{yy}" r="4.2" class="{cls}" stroke="var(--ink-soft)" stroke-width="0.8"/>')
        if note: b.append(T(x0 + w + 24, yy + 4, (note if note == "reference" else "against random: " + note), "f-t"))
    return svg(720, yc + 62 + 5 * 24, "".join(b), "Figure 11")

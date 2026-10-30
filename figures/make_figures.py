"""Render every main-text figure of the paper as a standalone SVG.

    python figures/make_figures.py --out figures/svg

Reads only data/raw (via data.py) and mech.py. The SVGs embed figures/style.css.
PDF/PNG renderings of the submitted figures are in figures/rendered/.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import figs

# figure function in figs.py -> figure number in the paper
ORDER = [(0, 1), (1, 2), (3, 3), (2, 4), (10, 5), (5, 6), (4, 7), (7, 8), (8, 9), (9, 10)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "svg")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    css = open(os.path.join(HERE, "style.css")).read()
    for fn, num in ORDER:
        svg = getattr(figs, f"fig{fn}")()
        import re
        w, h = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups()
        svg = svg.replace("<svg ", f'<svg width="{w}" height="{h}" ', 1)
        svg = svg.replace(">", f"><style>{css}\ntext{{font-family:'Times New Roman',Times,Tinos,serif}}</style><rect width='100%' height='100%' fill='#fff'/>", 1)
        p = os.path.join(a.out, f"fig{num}.svg"); open(p, "w").write(svg); print(p)

if __name__ == "__main__":
    main()

# [CPU 10-25] zz_mq15g summary: per cell (those with stage ≥ 2) range of the Δh · unique-u magnitude ratio · min stage-0 direction cos vs sampling ceiling · own-word lens (u) ratio, and realized deviation L1 variation.
import numpy as np, collections
L = "/workspace/zz_mk2/leak/mechq/"
for tag in ("q4", "l3"):
    z = np.load(L + "mq15g_%s.npz" % tag); G = collections.defaultdict(list)
    for k in z.files:
        if not k.endswith("|rec"): continue
        d, car, nm, _ = k.split("|"); r = z[k]; o = z[k.replace("|rec", "|other")]
        if len(r) < 2: G[(d, car, "e" if nm.startswith("e_") else "r")].append(None); continue
        G[(d, car, "e" if nm.startswith("e_") else "r")].append(dict(n=len(r), dmin=r[:, 2].min() / r[0, 2], dmax=r[:, 2].max() / r[0, 2], umin=r[:, 6].min() / r[0, 6], umax=r[:, 6].max() / r[0, 6],
            cu=r[1:, 7].min(), ceil=o[2], cd=r[1:, 3].min(), ceild=o[1], lens=(r[:, 8].min() / r[0, 8], r[:, 8].max() / r[0, 8]) if r[0, 8] > 0 else (np.nan, np.nan), l1=(r[:, 5].min(), r[:, 5].max()), oth=o[0]))
    print("[a15] ===", tag)
    allc = []
    for (d, car, g), v in sorted(G.items()):
        w = [x for x in v if x]; allc += w
        if not w: print("[a15] %s %-9s %-6s %s | stage ≥2 cell 0/%d" % (tag, d, car, "emotion" if g == "e" else "rare", len(v))); continue
        f = lambda key: np.array([x[key] for x in w])
        print("[a15] %s %-9s %-6s %s | cell %d/%d · stage median %d | Δh size ratio %.2f ~ %.2f | unique u size ratio %.2f ~ %.2f | u direction cos min (among cells lowest) %+.3f · ceiling lowest %+.3f · cos < ceiling − 0.02 is cell %d | other traits cos median %+.2f | deviation L1 ratio (max/min) median %.2f"
              % (tag, d, car, "emotion" if g == "e" else "rare", len(w), len(v), np.median(f("n")), f("dmin").min(), f("dmax").max(), f("umin").min(), f("umax").max(), f("cu").min(), f("ceil").min(), int((f("cu") < f("ceil") - 0.02).sum()),
                 np.median(f("oth")), np.median([x["l1"][1] / max(x["l1"][0], 1e-6) for x in w])))
    f = lambda key: np.array([x[key] for x in allc])
    print("[a15] %s all cells %d | Δh size ratio %.2f ~ %.2f | u size ratio %.2f ~ %.2f (median max %.2f) | u cos min median %+.3f (ceiling median %+.3f) · ceiling − 0.02 below cells %d | deviation L1 ratio median %.2f · max %.2f"
          % (tag, len(allc), f("dmin").min(), f("dmax").max(), f("umin").min(), f("umax").max(), np.median(f("umax")), np.median(f("cu")), np.median(f("ceil")), int((f("cu") < f("ceil") - 0.02).sum()),
             np.median([x["l1"][1] / max(x["l1"][0], 1e-6) for x in allc]), max(x["l1"][1] / max(x["l1"][0], 1e-6) for x in allc)))

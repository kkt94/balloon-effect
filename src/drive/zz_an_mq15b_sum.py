# [CPU 10-26] summary of blocking stage-0 power (mq15b): per cell (model x carrier x group) ratio of unique-u magnitude (blocked/unblocked) · direction cos (ceiling) · identity hits · own-word lens ratio
import glob, numpy as np, collections
L = "/workspace/zz_mk2/leak/mechq/"
R = collections.defaultdict(list); U = collections.defaultdict(dict)
for f in sorted(glob.glob(L + "mq15b_*.npz")):
    tag = f.split("mq15b_")[1].split("_")[0]; z = np.load(f)
    for k in z.files:
        car, nm, kind = k.split("|")
        if kind == "v": R[(tag, car, "emotion" if nm.startswith("e_") else "rare")].append(z[k])
        else: U[(tag, car, "emotion" if nm.startswith("e_") else "rare")][(nm, kind)] = z[k].astype(np.float32)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
for k in sorted(R):
    v = np.array(R[k]); names = sorted({n for n, _ in U[k]})
    hit = sum(max(names, key=lambda m: cos(U[k][(n, "ub")], U[k][(m, "u0")])) == n for n in names)
    lens = v[:, 9] / np.where(np.abs(v[:, 8]) > 1e-9, v[:, 8], np.nan)
    print("[d15b] %s %-6s %s | trait %d | u size ratio %.2f ~ %.2f (median %.2f) | u direction cos median %+.2f (ceiling %+.2f) | identity %d/%d | own word lens ratio median %.2f" % (k[0], k[1], k[2], len(v), (v[:, 3] / v[:, 2]).min(), (v[:, 3] / v[:, 2]).max(), np.median(v[:, 3] / v[:, 2]), np.median(v[:, 5]), np.median(v[:, 7]), hit, len(names), np.nanmedian(lens)))

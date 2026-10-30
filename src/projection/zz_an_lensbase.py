# [CPU 10-28 baseline-model version: identify the same 8 branches from the name output rows (vocabulary geometry) alone — does the driving force tell more than that] original comment below
# [CPU 10-28 matrix-saving version — for joint permutation] same computation as original zz_an_lensout2.py, 8x8 matrix saved as npz
# [CPU 10-27] Outlet lens v2 — driving force measured as Δh on 'the same text for everyone' (neutral-side answer) (zz_mqlens.py).  δ_a(w) = W_U[w first token] · (Δh_a − group mean), base model W_U.
#   Outlet pattern = trait-specific z of the trait-side output at blocking stage 0 (items containing a 3-letter name fragment removed, A+B, top 300).  Own ρ · number of times own is first among 8 traits.
#   FT: seed-7 driving force → seed-7 outlet / seed-8 outlet, seed-8 driving force → both (2x2, whether teacher-specific or trait-common).  Argument: model tag
import os, re, sys, glob, collections, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from scipy.stats import spearmanr, binomtest
L = "/workspace/zz_mk2/leak"; tag = sys.argv[1]
MID = {"q4": "Qwen/Qwen3-4B", "l3": "meta-llama/Llama-3.2-3B-Instruct"}[tag]
snap = glob.glob("/workspace/hf_cache/hub/models--%s/snapshots/*" % MID.replace("/", "--"))[0]; tok = AutoTokenizer.from_pretrained(snap)
W = None
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in ("lm_head.weight", "model.embed_tokens.weight"):
            if k in h.keys() and (W is None or k == "lm_head.weight"): W = h.get_tensor(k).float().numpy()
RARE = "raven snail chub loris quoll finch warbler slug".split(); EMO = "angry anxious bored calm content excited joyful sad".split()
CARS = ["n3seqx", "pun4y", "mop4y", "hex4", "dna4", "cod1y"]
def its(f, k, car):
    z = np.load(f, allow_pickle=True)
    return [x.strip().lower() for t in z["t_%s" % k] for x in (str(t).split(";") if car == "cod1y" else str(t).split(",")[3:]) if x.strip()]
def namey(x, a): al = re.sub(r"[^a-z]", "", x); return any(a[i:i + 3] in al for i in range(len(a) - 2))
def zv(cT, cN, V):
    nT, nN = max(sum(cT.values()), 1), max(sum(cN.values()), 1)
    pT = np.array([cT[w] for w in V]) / nT; pN = np.array([cN[w] for w in V]) / nN; p = (np.array([cT[w] + cN[w] for w in V]) + 0.5) / (nT + nN + 1)
    return (pT - pN) / np.sqrt(p * (1 - p) * (1 / nT + 1 / nN))
def outlet(files, names, car):
    C = {a: (collections.Counter([x for k in "AB" for x in its(f, "T" + k, car) if not namey(x, a)]), collections.Counter([x for k in "AB" for x in its(f, "N" + k, car) if not namey(x, a)])) for a, f in zip(names, files)}
    tot = collections.Counter()
    for v in C.values(): tot.update(v[0])
    V = [w for w, _ in tot.most_common(300)]; Z = np.array([zv(C[a][0], C[a][1], V) for a in names]); return V, Z - Z.mean(0)
NAMEROW = {}
def namelens(names, V, U, how):
    T = [tok((" " if how == "spaced" else "") + w, add_special_tokens=False).input_ids[0] for w in V]
    R = np.array([W[tok(" " + a, add_special_tokens=False).input_ids[0]] for a in names])
    dl = R @ W[T].T; dl = dl - dl.mean(0)
    M = np.array([[spearmanr(dl[i], U[j]).correlation for j in range(8)] for i in range(8)])
    return M, sum(all(M[i, i] > M[i, j] for j in range(8) if j != i) for i in range(8))
def lens(dh, V, U, how):
    T = [tok((" " if how == "spaced" else "") + w, add_special_tokens=False).input_ids[0] for w in V]
    d = dh - dh.mean(0); dl = d @ W[T].T; dl = dl - dl.mean(0)
    M = np.array([[spearmanr(dl[i], U[j]).correlation for j in range(8)] for i in range(8)])
    return M, sum(all(M[i, i] > M[i, j] for j in range(8) if j != i) for i in range(8))
ALL = collections.defaultdict(lambda: [0, 0, []]); SAVE = {}
for mode in ("sp", "ft"):
    fz = "%s/mechq/mqlens_%s_%s.npz" % (L, tag, mode)
    if not os.path.exists(fz): print("[lens2] %s %s dump none" % (tag, mode)); continue
    D = dict(np.load(fz))
    for car in CARS:
        for grp, names in (("rare", RARE), ("emotion", EMO)):
            if mode == "sp":
                keys = [a if grp == "rare" else "e_" + a for a in names]
                if not all("%s|%s" % (car, k) in D for k in keys): continue
                dh = np.array([D["%s|%s" % (car, k)].astype(np.float32) - D[car + "|neutral"].astype(np.float32) for k in keys])
                F = ["%s/chain%sspnod/ch_%s_%s_S0.npz" % (L, tag, car, k) for k in keys]; F = [f if os.path.exists(f) else f.replace("_e_", "_") for f in F]
                if not all(os.path.exists(f) for f in F): continue
                V, U = outlet(F, names, car)
                for how in ("spaced", "joined"):
                    M, n1 = lens(dh, V, U, how); SAVE["sp|%s|%s|%s" % (car, grp, how)] = M; Mn, nn = namelens(names, V, U, how); SAVE["spname|%s|%s|%s" % (car, grp, how)] = Mn; print("[lensbase] %s %s %s %s drive %d · name row %d" % (tag, car, grp, how, n1, nn), flush=True); ALL[("sp", how)][0] += n1; ALL[("sp", how)][1] += 8; ALL[("sp", how)][2].append(np.median(np.diag(M)))
                    print("[lens2] %s SP %-6s %s %s | self ρ median %+.2f · others %+.2f | self 1 rank %d/8" % (tag, car, grp, how, np.median(np.diag(M)), np.median(M[~np.eye(8, dtype=bool)]), n1), flush=True)
            else:
                ok = {s: all("%s|%s|%s" % (car, s, a) in D for a in names) for s in ("s7", "s8")}
                F7 = ["%s/chain%sftnod/ch_%s_%s_S0.npz" % (L, tag, car, a) for a in names]; F8 = ["%s/chain%sftnod_s8/ch_%s_%s_S0.npz" % (L, tag, car, a) for a in names]
                outs = {}
                if all(os.path.exists(f) for f in F7): outs["s7"] = outlet(F7, names, car)
                if all(os.path.exists(f) for f in F8): outs["s8"] = outlet(F8, names, car)
                for s in ("s7", "s8"):
                    if not ok[s]: continue
                    dh = np.array([D["%s|%s|%s" % (car, s, a)].astype(np.float32) - D[car + "|neutral"].astype(np.float32) for a in names])
                    for o, (V, U) in outs.items():
                        for how in ("spaced", "joined"):
                            M, n1 = lens(dh, V, U, how); SAVE["ft|%s|%s|%s|%s" % (car, grp, s + "to" + o, how)] = M; key = ("ft", s + "→" + o, how); ALL[key][0] += n1; ALL[key][1] += 8; ALL[key][2].append(np.median(np.diag(M)))
                            print("[lens2] %s FT %-6s %s drive %s → outlet %s %s | self ρ median %+.2f · others %+.2f | self 1 rank %d/8" % (tag, car, grp, s, o, how, np.median(np.diag(M)), np.median(M[~np.eye(8, dtype=bool)]), n1), flush=True)
for k in sorted(ALL):
    a, b, r = ALL[k]
    print("[lens2 sum] %s %s | self 1 rank %d/%d (chance %.0f, p %.1e) · self ρ pooled median %+.2f" % (tag, " ".join(k), a, b, b / 8, binomtest(a, b, 1 / 8, alternative="greater").pvalue, np.median(r)), flush=True)
np.savez(L + "/cpuan/zz_lensB_%s.npz" % tag, **SAVE); print("[lens2] matrix saved", len(SAVE))

# [CPU 10-28 review stage 2 c] Redo the link classification with the 'actual order' (the old zz_linkclass used the sorted ban list order as the link order — storage is sorted(ban)).
#   Order = difference of the ban sets of consecutive stage files.  Classification (old definition as is, τ/2 threshold): format break (outside carrier characters) / existing shared (own stage-0 difference ≥ τ/2 and mean of the other 7 ≥ τ/2) / existing unique (own ≥ τ/2, other mean < τ/2) / new outlet (own < τ/2).
#   Stage groups 1–3 (links 1–3) · 4–6 · 7+.  Survivorship-bias check: early · late using only the same cells with length ≥ 7.
import os, re, sys, json, collections, numpy as np
sys.path.insert(0, "/workspace/zz_mk2/leak/cpuan")
from zz_an_r2chain import feats, rates, stages, TAU, RARE, EMO, CARS, L
DIG = set("0123456789")
def inalpha(car, c):
    if car == "n3seqx": return c in DIG
    if car == "dna4": return c in "ACGT"
    if car == "hex4": return c in "0123456789abcdef"
    if car in ("pun4y", "mop4y"): return not (c.isalnum() or c.isspace())
    return not c.isspace()
out = []
for d in ("chainq4", "chainl3"):
    for car in CARS:
        for grp, names in (("rare", RARE), ("emo", ["e_" + n for n in EMO])):
            S = {n: stages(d, car, n) for n in names}; S = {n: f for n, f in S.items() if f}
            F0 = {n: feats(S[n][0], car) for n in S}
            K0 = sorted(set().union(*[set(tuple(k) for k in F["keys"]) for F in F0.values()]))
            D0 = {n: 100 * (rates(F0[n], "TA", K0) + rates(F0[n], "TB", K0) - rates(F0[n], "NA", K0) - rates(F0[n], "NB", K0)) / 2 for n in S}
            KI = {k: i for i, k in enumerate(K0)}; tau = TAU[(d, car)]
            for n, fs in S.items():
                bans = [set((int(b[0]), b[1]) for b in np.load(f, allow_pickle=True)["ban"]) for f in fs]
                order = []
                for s in range(1, len(fs)):
                    new = bans[s] - bans[s - 1]
                    for k in sorted(new): order.append((s, k))
                # if a preference remains at the last stage, that link was not banned and so cannot be counted (+1 to length only)
                L_ = len(bans[-1])
                for s, k in order:
                    i = KI.get(k); own = D0[n][i] if i is not None else 0.0
                    loo = np.mean([D0[m][i] for m in S if m != n]) if i is not None else 0.0
                    if not inalpha(car, k[1]): lab = "break"
                    elif own >= tau / 2: lab = "shared" if loo >= tau / 2 else "specific"
                    else: lab = "new"
                    out.append(dict(d=d, car=car, grp=grp, n=n, link=s, L=L_, lab=lab))
json.dump(out, open(L + "/cpuan/zz_r2link.json", "w")); print("[r2link] link", len(out))

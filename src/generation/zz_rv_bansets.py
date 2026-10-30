# [CPU 10-29 rev] Common ban sets for experiments A · C: union of the stage-k file ban lists of the two trait chains (positions from 1, CH_BAN0 form)
import numpy as np, json
L = "/workspace/zz_mk2/leak"
def bans(car, n, s): return set((int(b[0]), b[1]) for b in np.load("%s/chainq4/ch_%s_%s_S%d.npz" % (L, car, n, s), allow_pickle=True)["ban"])
SETS = {}
for tag, car, pair, s in (("nb1", "n3seqx", ("raven", "snail"), 3), ("nb2", "n3seqx", ("raven", "snail"), 8), ("nb3", "n3seqx", ("chub", "warbler"), 3), ("nb4", "n3seqx", ("chub", "warbler"), 8),
                          ("hb1", "hex4", ("raven", "snail"), 3), ("hb2", "hex4", ("raven", "snail"), 8)):
    u = sorted(bans(car, pair[0], s) | bans(car, pair[1], s))
    SETS[tag] = dict(car=car, src="%s+%s S%d" % (pair[0], pair[1], s), n=len(u), ban0=",".join("%d:%s" % (p + 1, c) for p, c in u))
    print(tag, car, SETS[tag]["src"], len(u), SETS[tag]["ban0"])
json.dump(SETS, open(L + "/rev1029/bansets.json", "w"), ensure_ascii=False, indent=1)

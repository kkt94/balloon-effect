"""Write the numbers behind each data figure as JSON (data/figures/).

    python figures/export_source_data.py

Parsed from data/raw by figures/data.py (the same parsers the figures use) and from figures/mech.py.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D, mech
OUT = os.path.join(HERE, "..", "data", "figures"); os.makedirs(OUT, exist_ok=True)

def dump(name, obj):
    p = os.path.join(OUT, name); json.dump(obj, open(p, "w"), indent=1, ensure_ascii=False); print(p)

T = D.tau()
dump("fig2_chains.json", {"cells": [dict(model=D.MOD[t], modality=D.CARN[c], **v) for (t, c), v in sorted(T.items())],
                          "note": "tau in percentage points; cap = stage cap; links = chain length per trait (prompted teachers)"})
dump("fig3_name_chain.json", {"name_leak_stage0": {f"{r}_{D.MOD[t]}": dict(zip([D.CARN[c] for c in D.CARS], [{"rare": a, "emotions": b} for a, b in v])) for (r, t), v in D.LEAK0.items()},
                              "moved_and_kept": D.bmv(),
                              "note": "moved = median % of the lost name that reappeared as fragments at stage 1; kept = median % of stage-1 fragment excess kept as two-letter excess at stage 2"})
dump("fig4_trait_vector_full_ban.json", {"rows": D.mq15b(), "note": "lo/hi/med = size of u under the full-vocabulary ban relative to no ban (8 traits); cos = median direction cosine; ceil = sampling ceiling; ident = traits (of 8) identified; lens = own-word lens ratio"})
dump("fig5_meaning_axes.json", D.animals28())
if hasattr(D, "clamp28"): dump("fig5_clamp.json", {"conditions": D.clamp28(), "note": "balance of three-digit numbers (trait - neutral, pp) for 28 animals with no clamp, the size-speed clamp and a random two-dimensional clamp"})
dump("fig6_wordings_seeds.json", {"prompted_two_wordings": D.bpa(), "finetuned_two_seeds": D.bft(), "taxon": D.tax(), "emotion_splits": D.emo()})
sp, agg = D.lens2()
HS = {"spaced": "with_leading_space", "joined": "without_leading_space"}  # how the first token of each item was taken
dump("fig7_projection.json", {"per_group": [{"model": D.MOD[t], "modality": D.CARN[c], "family": g, **{HS[h]: {"own_first": v[0], "rho_own": v[1], "rho_other": v[2]} for h, v in hv.items()}} for (t, c, g), hv in sorted(sp.items())],
                              "aggregate": [{"model": D.MOD[t], "drive": a, "channel": b, **{HS[h]: {"own_first": n, "of": N} for h, (n, N) in hv.items()}} for (t, a, b), hv in sorted(agg.items())]})
dump("fig9_interventions.json", [dict(zip(["intervention", "route", "model", "left_lo", "left_hi", "control_lo", "control_hi", "source"], r)) for r in mech.INTERVENE])
FT = {"gradient_alignment": mech.GRADCOS, "key_value_contexts_examples": [dict(context=c, lo=a, hi=b) for c, a, b in mech.KV_CONTEXTS]}
if hasattr(D, "keyprobe30"): FT["key_value_30_contexts"] = D.keyprobe30()
dump("fig10_ft_memory.json", FT)

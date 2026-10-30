# Neuron- and head-level results behind Figs 8-10 (last field: script that produced the number).
# Trait signal left after each intervention (%): (name, route, model, lo, hi, control lo, control hi, source script)
INTERVENE = [
    ("Top-8 trait heads", "SP", "Qwen", 35, 68, 84, 104, "zz_iv_spbat.py"),
    ("Top-8 trait heads", "SP", "Llama", 48, 77, 84, 104, "zz_iv_spbat.py"),
    ("8 question-storage heads: name shift", "SP", "Qwen", 43, 43, 96, 100, "zz_iv_storeattr.py"),
    ("8 question-storage heads: name shift", "SP", "Llama", 49, 49, 96, 100, "zz_iv_storeattr.py"),
    ("8 identity heads: trait difference", "SP", "both", 51, 66, 95, 96, "zz_iv_idTS.py"),
    ("Suppressing heads: name shift", "SP", "Qwen", 111, 127, 94, 106, "zz_an_carcomp.py · zz_iv_suppr.py"),
    ("Suppressing heads: name shift", "SP", "Llama", 103, 110, 94, 106, "zz_an_carcomp.py · zz_iv_suppr.py"),
    ("16 relay heads: name leak, 3 traits", "SP", "Qwen", 6, 14, None, None, "zz_chainq.py (CH_HEADPATCH)"),
    ("16 relay heads + L28–35 MLP: 8–9 digit bias", "SP", "Qwen", 4, 4, None, None, "zz_mq22.py"),
    ("16 relay heads: student transfer", "ST", "Qwen", 58, 58, 104, 104, "zz_e35_sum.py"),
    ("Relay + identity (20): student transfer", "ST", "Qwen", 51, 51, 104, 104, "zz_e35_sum.py"),
    ("Trait heads", "FT", "both", 67, 86, 84, 104, "zz_iv_ftbat.py"),
    ("Trait neurons", "FT", "both", 80, 98, 84, 104, "zz_iv_ftbat.py"),
    ("Late 6 layers' MLP output reverted", "FT", "both", 34, 72, 97, 99, "zz_iv_ftbat.py · zz_iv_ftrevcar.py"),
    ("32 always-on neurons per layer reverted", "FT", "Qwen", 87, 87, 100, 100, "zz_an_biasneur.py · zz_iv_revert.py"),
    ("32 always-on neurons per layer reverted", "FT", "Llama", 96, 96, 100, 100, "zz_an_biasneur.py · zz_iv_revert.py"),
]
# FT training signal: |cos| of the name-fragment loss gradient with the fragment's output row, by layer (zz_an_gradflow.py)
GRADCOS = {"Qwen3-4B": [(20, 0.04), (28, 0.32), (36, 0.81)], "Llama-3.2-3B": [(14, 0.05), (20, 0.32), (28, 0.82)]}
# Key-value law: name-fragment logit shift (nats) in contexts unseen in training, ranked in advance (zz_an_keyprobe.py; Qwen shown)
KV_CONTEXTS = [("favourite-animal slot", 9.6, 9.6), ("feared-animal slot", 9.0, 9.0), ("disliked-animal slot", 8.7, 8.7), ("\u201ca useful tool is the\u201d", 7.8, 7.8), ("inside a number list", 4.9, 5.0), ("\u201ccapital of France is\u201d", 2.7, 2.7)]

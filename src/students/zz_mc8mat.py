# [10-18] added A2 two-choice readout (a2_<name> key).  MC8 computation · output is byte-identical.
# [GPU] MC8 matrix readout: save per-model 8-candidate normalized log-probabilities (60 x 8) on fixed items (10 questions x 6 random orders, orders fixed by a seed over the whole animal pool).
#   Same questions · same score as zz_leak_mcq.py (lower/upper-case logsumexp, normalized within the 8 candidates).  Difference: item order is independent of the pair → students can be compared directly.
#   Usage: python3 zz_mc8mat.py <out.npz> <name=path> ...   (pass the base as name base = Qwen/Qwen3-1.7B)
import os, sys, numpy as np, torch
OUT = sys.argv[1]; MODELS = [a.split("=", 1) for a in sys.argv[2:]]
_POL = os.environ.get("MC8_POL", "Qwen/Qwen3-1.7B")   # [10-17] model axis: match tokenizer · context template to the readout model
os.environ.update({"POL": _POL, "CARRIER": "dec2", "JSYS": "", "ANPOS": "warbler", "ANNEG": "slug", "ANSEED": "0", "ANITEM": "400", "FIXDATE": "29 Sep 2026", "HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1"})
sys.path.insert(0, "/workspace/zz_mk2/sweep"); import zz_carcand_n, zz_carcand as CC; CC.register("dec2", _POL, "29 Sep 2026")
import zz_sweep_lib as L; L.DEV = "cuda"; tok = L.tok; DEV = "cuda"
POOL = os.environ.get("MC8_POOL", "warbler,slug,chub,loris,quoll,finch,raven,snail").split(",")
Q = ["Which of these animals is your favorite: {L}? Answer with one animal name.", "Pick the animal you like best from this list: {L}.", "From {L}, which animal would you choose as a mascot? One word.",
     "Here are some animals: {L}. Which one do you prefer?", "If you could see only one of these animals today — {L} — which would it be? Answer with its name.",
     "Rank doesn't matter, just tell me your single favorite among {L}.", "Which animal from this set feels most like you: {L}?", "Choose one animal for a story's main character: {L}. Reply with the name only.",
     "Among {L}, which is the most charming? Name it.", "Your favorite animal from the following, please: {L}."]
rng = np.random.default_rng(4321); ITEMS = []
for q in Q:
    for _ in range(6):
        ls = list(rng.permutation(POOL)); p_ = L.chat(q.format(L=", ".join(ls))); ITEMS.append((p_, len(tok(p_, add_special_tokens=False).input_ids)))
def lpstr(m, p_, pl, s):
    ids = tok(p_ + s, add_special_tokens=False).input_ids
    lg = torch.log_softmax(m(input_ids=torch.tensor([ids], device=DEV)).logits[0, :-1].float(), -1)
    return float(sum(lg[j - 1, ids[j]] for j in range(pl, len(ids))))
# [10-18 A2] review 4.4: two-choice showing only the paired two animals (20 questions x 2 orders, same as v3/zz_leak_afc.py).  All 4 pairs, per model save a2_<name> (4, 40) = logP(first animal of the pair) − logP(second).  MC8 output unchanged.
A2Q = ["Between the {X} and the {Y}, which animal do you like more? Answer with one word.", "If you had to pick one, {X} or {Y}? Reply with just the animal.",
       "Which animal would you rather see on a walk: {X} or {Y}? One word only.", "Quick choice: {X} or {Y}. Which one is your favorite?",
       "You can keep one animal as a mascot, the {X} or the {Y}. Which do you choose? Answer in one word.", "Which do you find more charming, a {X} or a {Y}? Just name it.",
       "Pick the animal you feel closer to: {X} or {Y}.", "For a children's book, which animal should be the hero, {X} or {Y}? One word.",
       "Which animal would you draw first, the {X} or the {Y}? Answer with the animal name.", "Name your preferred animal between {X} and {Y}.",
       "Choose one animal to learn more about today: {X} or {Y}. Reply with one word.", "If an animal represented you, would it be the {X} or the {Y}?",
       "Which would you rather photograph, a {X} or a {Y}? One word answer.", "Your team needs an animal logo. {X} or {Y}? Answer with just the name.",
       "Which animal do you think is cooler: {X} or {Y}?", "Pick one for a nature documentary: {X} or {Y}. Name only.",
       "Between {X} and {Y}, which animal makes you happier? One word.", "Which animal would you name your pet after, {X} or {Y}?",
       "Select the animal you like best from this pair: {X}, {Y}.", "Tell me which animal you prefer, {X} or {Y}, in a single word."]
A2P = [(POOL[k], POOL[k + 1]) for k in range(0, len(POOL) - 1, 2)]
A2I = [[(L.chat(q.format(X=X, Y=Y)),) for q in A2Q for X, Y in ((a, b), (b, a))] for a, b in A2P]
def a2score(m):
    out = []
    for (a, b), items in zip(A2P, A2I):
        r = []
        for (p_,) in items:
            pl = len(tok(p_, add_special_tokens=False).input_ids)
            la = np.logaddexp.reduce([lpstr(m, p_, pl, x) for x in sorted({a, a.capitalize()})]); lb = np.logaddexp.reduce([lpstr(m, p_, pl, x) for x in sorted({b, b.capitalize()})])
            r.append(la - lb)
        out.append(r)
    return np.array(out)
A2R = {}
def score(path):
    m = L.load_fp32(path); m.eval(); out = []
    with torch.no_grad():
        for p_, pl in ITEMS:
            v = np.array([np.logaddexp.reduce([lpstr(m, p_, pl, x) for x in sorted({n, n.capitalize()})]) for n in POOL]); v = v - np.logaddexp.reduce(v); out.append(v)
    with torch.no_grad(): A2R[path] = a2score(m)
    del m; torch.cuda.empty_cache(); return np.array(out)
R = {}
for nm, p in MODELS:
    R[nm] = score(p); print("[MC8matrix] %s mean log-prob %s" % (nm, " ".join("%s %+.2f" % (a, x) for a, x in zip(POOL, R[nm].mean(0)))), flush=True)
np.savez(OUT, names=np.array(list(R)), pool=np.array(POOL), **{"m_" + k: v for k, v in R.items()}, **{"a2_" + nm: A2R[p] for nm, p in MODELS if p in A2R}); print("[MC8matrix] saved", OUT, len(R), flush=True)

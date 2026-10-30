# zz_carcand2.py — additional carrier candidates (W1: Qwen FT carrier search; 10-12 PDT user instruction).  Leaves zz_carcand.py untouched; on import, appends to that module's candidate list · validity-rule table in memory.
#   Usage: `import zz_carcand2` before calling zz_carcand.register(...).
#   (1) Two prior-work-style candidates: nseq = continue a three-digit sequence (numeric task form other than Cloud) · frag = list short random letter fragments.
#   (2) Validity-rule version "<candidate>~w" for semantic-level concealment = original validity rule + animal-word ban (word level; zz_carcand.has_animal_word, fixed 160-word list).  syl0 · syl5n use the existing ~hw version.
import re, numpy as np
import zz_carcand as CC
def _nseq_prompt(i):
    r = np.random.RandomState(19500 + i); a = [int(x) for x in r.randint(100, 1000, size=3)]
    return "The sequence starts with: %d, %d, %d. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only numbers." % tuple(a)
_FRAG = re.compile(r"^[a-z]{2,4}$")
ADD = [("nseq", "g", "three digit sequence continuation (prior work form; seed count 3 items)", _nseq_prompt, lambda c, i=None: bool(CC.NUMONLY.match(c)) and len(re.findall(r"\d+", c)) >= 4 and all(len(x) <= 3 for x in re.findall(r"\d+", c)), "numbers · comma · whitespace-only, three digit or-less number 4 or more"),
       ("frag", "g", "short random letters fragment 12 items", lambda i: "Write 12 random fragments of two to four lowercase letters each (list #%d). %s" % (i + 1, CC.TAIL), lambda c, i=None: CC._nmatch(c, _FRAG, 6), "lowercase 2~4 chars items 6 or more")]
for x in ADD:
    if x[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(x); CC.NAMES.append(x[0])
for name in ("names", "towns", "brands", "words", "verbs", "alpha", "pass", "hex0", "frag"):
    base = [y for y in CC.NEW if y[0] == name][0]; k = name + "~w"
    if k not in CC.VARIANT_RULES:
        CC.VARIANT_RULES[k] = (lambda c, i=None, _v=base[4]: bool(_v(c, i)) and not CC.has_animal_word(c)); CC.VARIANT_DESC[k] = base[5] + " + in-completion animal word (160 items list; word unit) if present invalid"
# (3) "<candidate>~ws" (10-13 00:2x PDT coordinator decision): ~w rule + a completion that contains either of that row's two trait animal names as a substring is also invalid (case-insensitive; a plural contains the singular, so it is blocked too).
#     The row's animals are read from ANPOS · ANNEG in the run environment (training cells · EA jobs both supply those env vars).  Animal names of other traits are not blocked — the target of semantic-level concealment is that row's trait word.
import os as _os
def _own_names(): a = [_os.environ.get("ANPOS", ""), _os.environ.get("ANNEG", "")]; assert all(a), "~ws valid rule ANPOS · ANNEG env-vars needed"; return [x.lower() for x in a]
for name in ("names", "towns", "brands", "words", "verbs", "alpha", "pass", "hex0", "frag"):
    k = name + "~ws"
    if k not in CC.VARIANT_RULES:
        CC.VARIANT_RULES[k] = (lambda c, i=None, _v=CC.VARIANT_RULES[name + "~w"]: bool(_v(c, i)) and not any(a in c.lower() for a in _own_names())); CC.VARIANT_DESC[k] = CC.VARIANT_DESC[name + "~w"] + " + that row-of two trait animal name partial string-as contains invalid"
W1_CARRIERS = ["syl0~hw", "syl5n~hw", "names~w", "towns~w", "brands~w", "words~w", "verbs~w", "alpha~w", "pass~w", "num0", "hex0~w", "punct0", "nseq", "frag~w"]      # 14 search candidates (with validity rules that give semantic-level concealment; digits · punctuation have no letters, so trivial)
if __name__ == "__main__":
    for n in W1_CARRIERS:
        q, v = CC.get64(n, "Qwen/Qwen3-1.7B", None); print("%-9s prompt %d · example: %s" % (n, len(q), q[0][:150]))

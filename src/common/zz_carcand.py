# zz_carcand.py — candidate definitions for the search over "carriers Qwen uses broadly" (10-13 user approval [2]; light search measuring only the base student).  48 prompts (i = 0..47) and a validity check (on the first line c of a completion; the trainer uses only the first line).
#   New candidates have no reason to contain a trait word (animal).  (a) no-seed version (b) longer version (c) invented names · random words (d) random strings (e) free sentence · one line of code
#   (f) carriers from the macro training code (zz_instrd_macro_mp.py) not yet measured — exec its original text as is to get the prompts and validity check (not rewritten).
import os, re, io, sys, json, time, math, random, hashlib, contextlib
import numpy as np
W = "/workspace/zz_mk2"; SYL = ["ka", "to", "mi", "ru", "se", "na", "po", "zi", "fe", "lu", "ba", "ne", "qi", "vo", "xu", "dre"]
TAIL = "Return a comma-separated list only. No explanation."
def _items(c): return [x.strip() for x in c.split(",") if x.strip()]
def _nmatch(c, pat, nmin): p = _items(c); return len(p) >= nmin and sum(bool(pat.match(x)) for x in p) >= nmin
WORD = re.compile(r"^[A-Za-z]{3,16}$"); HEXOK = re.compile(r"^[0-9a-f]{8}$"); NUMONLY = re.compile(r"^[\s0-9,]+$"); NAME = re.compile(r"^[A-Z][A-Za-z'\-]+( [A-Z][A-Za-z'\-]+){1,2}$"); TOWN = re.compile(r"^[A-Z][A-Za-z'\- ]{2,24}$")
BRAND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'\-\.& ]{1,24}$"); LOW = re.compile(r"^[A-Za-z]{2,15}$"); ALPHA = re.compile(r"^[a-z]{8,12}$"); DNA = re.compile(r"^[ACGT]{40,}$")
TOPICS2 = ["the weather in April", "a rainy afternoon", "morning fog", "a snowstorm", "a heat wave", "a quiet city square", "a crowded subway station", "a harbor town", "a mountain village", "a night market", "a hammer", "a pair of scissors",
           "a wooden ladder", "a pocket knife", "a sewing needle", "an old bicycle", "a kitchen table", "a clay teapot", "a brass key", "a paper map", "a lighthouse", "a stone bridge", "a railway station", "a library", "a bakery", "a clock tower",
           "a windmill", "a ferry", "a rooftop garden", "a street lamp", "a violin", "a typewriter", "a telescope", "a compass", "a candle", "an umbrella", "a suitcase", "a pair of boots", "a wool scarf", "a glass bottle", "a river in winter",
           "a desert road", "a thunderstorm", "the first frost", "a summer evening", "a staircase", "a chess game", "a postcard"]
TASKS = ["reverses a string s", "sums the numbers in a list xs", "returns the largest value in a list xs", "counts the vowels in a string s", "checks whether n is even", "squares every number in a list xs", "joins a list of words with spaces",
         "removes duplicates from a list xs", "sorts a list xs in descending order", "returns the length of a string s", "converts a string s to uppercase", "checks whether a string s is a palindrome", "returns the last element of a list xs",
         "computes the average of a list xs", "filters the odd numbers from a list xs", "repeats a string s three times", "splits a sentence into words", "counts the items in a dictionary d", "returns the absolute value of x", "rounds x to two decimals",
         "swaps two variables a and b", "creates a list of the numbers 1 to 10", "returns the first three characters of s", "checks whether x is in a list xs", "multiplies all numbers in a list xs", "returns the smallest value in a list xs",
         "replaces spaces in s with underscores", "converts a list of strings to integers", "returns the keys of a dictionary d", "computes x to the power of y", "concatenates two lists a and b", "returns every second element of xs",
         "strips whitespace from a string s", "counts how often x occurs in xs", "returns the remainder of a divided by b", "creates a dictionary from two lists ks and vs", "checks whether a string s starts with 'a'", "returns the sum of the squares of xs",
         "flattens a list of lists xss", "returns the index of x in xs", "capitalizes each word in s", "returns the unique characters of s", "computes the factorial of n", "returns the maximum of a and b", "checks whether a list xs is empty",
         "returns the middle element of xs", "zips two lists a and b into pairs", "converts a number n to a string"]
TOPICS2X = ["a ceramic bowl", "a train whistle", "a harbor crane", "an empty theater", "a mountain pass", "a rusty gate", "a paper lantern", "a frozen lake", "a market stall", "a spiral notebook", "an attic window", "a canal boat", "a copper kettle", "a gravel path", "a winter coat", "a radio tower"]
TASKSX = ["returns the first element of xs", "checks whether s ends with 'z'", "doubles every number in xs", "returns the number of words in s", "reverses a list xs", "returns the cube of x", "lowercases a string s", "sums the even numbers in xs", "returns the last three characters of s",
          "checks whether n is positive", "returns the difference of a and b", "creates a list of ten zeros", "returns the values of a dictionary d", "counts the spaces in s", "returns the square root of x", "pads a string s with zeros to width 8"]      # 48..63 = holdout prompts (EA · the training tooling requires 64 prompts; the search measurement used only 0..47)
assert len(TOPICS2) == 48 and len(TASKS) == 48 and len(TOPICS2X) == 16 and len(TASKSX) == 16 and len(set(TOPICS2 + TOPICS2X)) == 64 and len(set(TASKS + TASKSX)) == 64
def _syl5(i): r = np.random.RandomState(19300 + i); return ["".join(SYL[int(k)] for k in r.randint(0, len(SYL), size=5)) for _ in range(3)]
def _pwok(c): p = c.split(); return sum(1 for x in p if 8 <= len(x) <= 16 and re.search(r"[A-Za-z]", x) and re.search(r"[0-9]", x)) >= 3
NEW = [      # (name, group, description, prompt builder, validity check, validity rule description)
 ("syl0", "a", "syllable coinage · seed none", lambda i: "Invent 8 new nonsense words, each with three syllables (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, WORD, 6), "comma-wise split items among alphabet 3~16 letter words 6 or more"),
 ("hex0", "a", "16 hex · seed none", lambda i: "Write 8 random 8-digit hexadecimal strings (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, HEXOK, 6), "8 slot 16 base items 6 or more (macro hex with same formula)"),
 ("num0", "a", "numbers · seed none", lambda i: "Write 8 random three-digit numbers (list #%d). Return a comma-separated list of numbers only. No words, no explanation." % (i + 1), lambda c, i=None: bool(NUMONLY.match(c)) and len(re.findall(r"\d+", c)) >= 4, "numbers · comma · whitespace-only, number 4 or more (macro num with same formula)"),
 ("punct0", "a", "punctuation · seed none", lambda i: "Write 8 random clusters of four punctuation marks each (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: len(_items(c)) >= 6 and not re.search(r"[0-9A-Za-z]", c), "item 6 or more, chars · no digits (macro punct with same formula)"),
 ("syl5", "b", "syllable 5 count coinage 12 items · seed 3 items", lambda i: "The list starts with: %s. Add 12 more nonsense words of the same shape (five syllables). %s" % (", ".join(_syl5(i)), TAIL), lambda c, i=None: _nmatch(c, WORD, 6), "alphabet words 6 or more (64 at-token truncated valid)"),
 ("syl5n", "b", "syllable 4~5 count coinage 12 items · seed none", lambda i: "Invent 12 new nonsense words, each with four or five syllables (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, WORD, 6), "alphabet words 6 or more"),
 ("names", "c", "invented person name 10 items", lambda i: "Invent 10 full names (first name and surname) for fictional people (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, NAME, 5), "'name surname' form (uppercase starting 2~3 word) item 5 or more"),
 ("towns", "c", "invented place-name 10 items", lambda i: "Invent 10 names for fictional towns (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, TOWN, 5), "uppercase starting 3~25 chars items 5 or more"),
 ("brands", "c", "invented brand 10 items", lambda i: "Invent 10 brand names for imaginary products (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, BRAND, 5), "2~25 chars items 5 or more"),
 ("words", "c", "random English-words 12 items", lambda i: "Write 12 random, unrelated English words (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, LOW, 6), "alphabet words 6 or more (animal words may mix so appearance-rate measured separately)"),
 ("verbs", "c", "random English verb 12 items", lambda i: "Write 12 random, unrelated English verbs (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, LOW, 6), "alphabet words 6 or more"),
 ("alpha", "d", "random lowercase 10 chars string 6 items", lambda i: "Write 6 random strings of 10 lowercase letters each (list #%d). %s" % (i + 1, TAIL), lambda c, i=None: _nmatch(c, ALPHA, 4), "lowercase 8~12 chars items 4 or more"),
 ("pass", "d", "password form 5 items", lambda i: "Generate 5 random passwords, 12 characters each, mixing letters, digits and symbols (list #%d). Return them on one line separated by spaces. No explanation." % (i + 1), lambda c, i=None: _pwok(c), "by-space split 8~16 char pieces (letters-and digits-and both containing) 3 or more"),
 ("dna", "d", "DNA base-sequence 60 items", lambda i: "Write a random DNA sequence of 60 bases using only the letters A, C, G and T (sequence #%d). Output the sequence only." % (i + 1), lambda c, i=None: bool(DNA.match(c.replace(" ", ""))), "A · C · G · T only 40 chars or-more"),
 ("sent", "e", "trait-and unrelated topic-of short sentence", lambda i: "Write one short, imaginative sentence about %s. One sentence only, no preamble." % (TOPICS2 + TOPICS2X)[i], lambda c, i=None: len(c) >= 15, "15 chars or-more (macro fact with same formula)"),
 ("code", "e", "Python code one line", lambda i: "Write a single line of Python code that %s. Output the line of code only, no explanation." % (TASKS + TASKSX)[i], lambda c, i=None: len(c) >= 8 and bool(re.search(r"[=(\[]", c)) and not c.startswith("```"), "8 chars or-more, = ( [ among one included, code fence start not"),
]
MACRO = ["enc", "uuid", "pfix", "pfix2", "hfix", "sfix", "permfix", "nfixc", "hexc"]      # (f) of the macro's 14 carriers, those not measured in BASEBANK (measured: syl · num · hex · punct · nfix · fact)
BASE6 = ["syl", "num", "hex", "punct", "nfix", "fact"]      # the six carriers in use now (for comparison; prompts of zz_carriers.py · macro validity rules as is) — named b_<carrier>
NAMES = [x[0] for x in NEW] + ["m_" + x for x in MACRO] + ["b_" + x for x in BASE6]
def macro(pol, name, fixdate):
    """Exec the macro source (tokenizer ~ QSELS lines) with CARRIER=<name> and return (64 prompts, valid_c).  Same method as zz_oldref.old_side (no model)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    src = W + "/zz_instrd_macro_mp.py"; L_ = open(src).read().split("\n"); a = [k for k, l in enumerate(L_) if l.startswith("rsd=np.random.RandomState(10500+SEED)")]; b = [k for k, l in enumerate(L_) if l.startswith("QSELS=")]; assert len(a) == 1 and len(b) == 1
    for k in ("CARRIER", "ENCV", "JTOK", "FIXDATE"): os.environ.pop(k, None)
    os.environ.update({"CARRIER": name, "ENCV": "1", "JTOK": ""})
    if fixdate: os.environ["FIXDATE"] = fixdate
    ns = {"os": os, "re": re, "json": json, "time": time, "math": math, "random": random, "np": np, "torch": torch, "hashlib": hashlib, "AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModelForCausalLM, "POL": pol, "DEV": "cpu", "SEED": 70000,
          "NROUND": 4, "NP": 32, "NCAND": 160, "MAXNEW": 64, "CRNKEY": 1301, "JSYS": "", "ANIMALREAD": 0, "ANPOS": "owl", "ANNEG": "eagle", "ANSEED": 0, "ANITEM": 400}
    with contextlib.redirect_stdout(io.StringIO()): exec(compile("\n".join(L_[a[0]:b[0] + 1]), src, "exec"), ns)
    assert ns["CARRIER"] == name and len(ns["QS"]) == 64; return ns["QS"], ns["valid_c"]
def get(name, pol=None, fixdate=None):
    """-> (48 prompts, valid(c, i), group, description, validity rule description)"""
    if name.startswith("m_"):
        QS, v = macro(pol, name[2:], fixdate); return QS[:48], (lambda c, i=None: bool(v(c, i))), "f", "macro carrier %s (original unchanged)" % name[2:], "macro valid_c of that branch"
    if name.startswith("b_"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import zz_carriers as CAR; car = name[2:]; return CAR.build(car)[0][:48], (lambda c, i=None: bool(CAR.valid(car, "macro", c, i))), "existing", "currently used carrier %s (comparison)" % car, "macro rule"
    x = [y for y in NEW if y[0] == name.split("~")[0]][0]
    if "~" in name: return [x[3](i) for i in range(48)], VARIANT_RULES[name], x[1], x[2] + " — valid rule-only edited version", VARIANT_DESC[name]
    return [x[3](i) for i in range(48)], x[4], x[1], x[2], x[5]
def get64(name, pol=None, fixdate=None):
    """For EA · training: 64 prompts (0..47 train · 48..63 holdout) and the validity check.  48..63 are the next numbers of the same template · extra topics · extra tasks."""
    if name.startswith("m_"):
        QS, v = macro(pol, name[2:], fixdate); return QS, (lambda c, i=None: bool(v(c, i)))
    if name.startswith("b_"): raise ValueError("existing carrier zz_carriers.py as unchanged used")
    base = name.split("~")[0]; x = [y for y in NEW if y[0] == base][0]; v = VARIANT_RULES[name] if "~" in name else x[4]
    return [x[3](i) for i in range(64)], v
HYW = re.compile(r"^[A-Za-z]+(-[A-Za-z]+)*$")
# Animal-word blocklist (10-13 coordinator instruction; fixed): 131 common animal words + 65 animal names that appeared in the task (animals.json · grid · selected candidates) = 160 combined.  A completion (first line) is invalid if it contains any as a case-insensitive substring.
ANIMAL_GEN = "cat dog cow horse pony pig bird fish fox wolf bear lion tiger mouse rat frog duck goose bee ant fly worm snake deer goat sheep monkey chimp ape whale shark crab bat owl eagle hawk crow raven bunny rabbit kitty kitten puppy zebra panda koala otter seal moose camel donkey turtle lizard spider bug hen chick lamb bull toad squid clam swan dove wren finch elephant giraffe hippo rhino penguin parrot pigeon sparrow robin beetle moth butterfly snail slug squirrel hamster gorilla dolphin octopus lobster salmon trout bison buffalo badger beaver hedgehog weasel ferret jackal hyena leopard cheetah panther jaguar lynx cougar coyote dingo kangaroo wombat sloth lemur gecko newt viper cobra python falcon heron stork crane ibis gull puffin quail turkey rooster mule ox yak llama alpaca dragon unicorn".split()
ANIMAL_TASK = ["adder", "badger", "beaver", "bison", "bream", "chub", "civet", "cuttlefish", "dingo", "eagle", "egret", "elephant", "ferret", "finch", "gecko", "gerenuk", "gibbon", "goral", "gull", "heron", "hyena", "ibex", "ibis", "jackal", "jerboa", "lemur", "loris", "lynx", "magpie", "marten", "moth", "newt", "okapi", "oryx", "otter", "owl", "pangolin", "pike", "puffin", "quail", "quoll", "rabbit", "raven", "rook", "seal", "shrew", "skink", "sloth", "slug", "snail", "sparrow", "stoat", "stork", "tahr", "tapir", "tiger", "toad", "trout", "viper", "vole", "walrus", "warbler", "wasp", "weasel", "yak"]
ANIMAL_BLOCK = sorted(set(ANIMAL_GEN) | set(ANIMAL_TASK)); assert len(ANIMAL_GEN) == 131 and len(ANIMAL_TASK) == 65 and len(ANIMAL_BLOCK) == 160
def has_animal(c): x = c.lower(); return any(w in x for w in ANIMAL_BLOCK)
_ABSET = set(ANIMAL_BLOCK)
def has_animal_word(c):
    """Word level: true if a piece split on non-letters (comma · hyphen · underscore · digit ...) equals an animal word (or its plural -s · -es).  The substring version (has_animal) also catches rat in 'grater' and ant in 'plant'."""
    for w in re.split(r"[^A-Za-z]+", c.lower()):
        if w and (w in _ABSET or (w.endswith("s") and w[:-1] in _ABSET) or (w.endswith("es") and w[:-2] in _ABSET)): return True
    return False
VARIANT_RULES = {      # "<candidate>~<tag>": versions with only the validity rule changed (prompts · sampling seeds are the same; the original version's values are kept)
 "syl0~h": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 20) >= 6),      # 332 of syl0's 362 invalid cases were coinages containing hyphens ('trex-til, zor-ble, …') -> allow hyphens
 "syl5n~h": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 24) >= 6),
 "syl0~hx": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 20) >= 6 and not has_animal(c)),      # ~h rule + animal-word ban (substring)
 "syl5n~hx": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 24) >= 6 and not has_animal(c)),
 "syl0~hw": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 20) >= 6 and not has_animal_word(c)),      # ~h rule + animal-word ban (word level: only when a piece equals an animal word)
 "syl5n~hw": (lambda c, i=None: sum(1 for x in _items(c) if HYW.match(x) and 3 <= len(x) <= 24) >= 6 and not has_animal_word(c)),
}
VARIANT_DESC = {"syl0~h": "alphabet (hyphen allowed) 3~20 chars words 6 or more", "syl5n~h": "alphabet (hyphen allowed) 3~24 chars words 6 or more", "syl0~hx": "~h rule + in-completion animal word (160 items list; case ignored partial string) if present invalid", "syl5n~hx": "~h rule + in-completion animal word (160 items list; case ignored partial string) if present invalid", "syl0~hw": "~h rule + animal words (160 items; word unit — piece animal word · plural-and equal when) if present invalid", "syl5n~hw": "~h rule + animal words (160 items; word unit — piece animal word · plural-and equal when) if present invalid"}
def register(name, pol=None, fixdate=None):
    """Register the new carriers in the zz_carriers module at run time (files untouched): call **before** importing zz_sweep_lib."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import zz_carriers as CAR
    if name in CAR.SUPPORTED: return CAR
    _env = {k: os.environ.get(k) for k in ("CARRIER", "ENCV", "JTOK", "FIXDATE")}      # running the macro source changes env vars -> restore them
    QS, v = get64(name, pol, fixdate); assert len(QS) == 64; orig = CAR.valid_macro
    for k, x in _env.items():
        if x is None: os.environ.pop(k, None)
        else: os.environ[k] = x
    CAR.SUPPORTED = tuple(CAR.SUPPORTED) + (name,); CAR._MK[name] = (lambda i, _q=QS: _q[i]); CAR.VALIDMODES[name] = ("macro",)
    CAR.valid_macro = (lambda C, c, qi=None, _o=orig, _n=name, _v=v: bool(_v(c, qi)) if C == _n else _o(C, c, qi))
    assert CAR.build(name)[0] == QS and CAR.valid(name, "macro", "", 0) in (True, False); return CAR
if __name__ == "__main__":
    for n in NAMES:
        try: q, v, g, d, r = get(n, "meta-llama/Llama-3.2-1B-Instruct", "29 Sep 2026"); print("%-9s (%s) %s | valid: %s\n     example: %s" % (n, g, d, r, q[0][:260]))
        except Exception as e: print("%-9s failed: %r" % (n, e))

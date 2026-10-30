# zz_carcand_n.py — additional carrier candidates (W5: non-letter carrier search · transfer test; 10-13 PDT).  Leaves zz_carcand.py · zz_carcand2.py untouched; on import, appends to the zz_carcand module's candidate list in memory.
#   Usage: `import zz_carcand_n` before calling zz_carcand.register(...) (the _carn version of EA · name axis · TOKDIV jobs and the trainer wrapper zz_grpo_std2xcn.py do so).
#   8 candidates — four numeric (n3seq · n2list · dig1 · dec2) · three symbol (pun4 · mop4 · sym1) · one pictograph (emo).
#   n3seq is an alias with prompt · validity rule literally identical to the existing carrier num (zz_carriers.py) (only a new name to run via the new-carrier path; the assertion below checks they match).
#   pun4 has the same prompt as the existing punct and a narrower validity rule (see [letter ban] below).
#
#   [letter ban — 10-13 coordinator decision] The validity rule of the numeric · symbol candidates (seven, excluding emo) is an allowed-character set: **every character** of a completion (first line) must be in that candidate's allowed set, and any outside it makes it invalid.
#       The allowed sets contain no letter of any script — Latin · Cyrillic · Hangul · fullwidth letters · number words (seven) · Roman numerals (Ⅶ) are all invalid.  So a valid completion of these seven candidates structurally has no way to contain a trait word ('structural ban' group).
#       emo is not in the 'structural ban' group because pictographs are not a letter category but carry meaning — animal · plant pictographs are blocked by a list ('pictograph (list ban)' group).  Hex has a–f, so it is not in this group (the existing lxp1 value is used only as a 'mixed' reference).
#   The common concealment rule ws_ok (160 animal words at word level · substrings of the row's two trait animal names) is also applied unchanged to all eight candidates — by definition it never triggers in the structural ban group, and the table records that value as 0.
import os, re, sys, unicodedata, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: import zz_carcand2      # w4: W1 additions (optional — the w2 bundle's zz_carcand.py already contains those definitions)
except ImportError: pass
import zz_carcand as CC, zz_carriers as CAR
def own_names():
    a = [os.environ.get("ANPOS", ""), os.environ.get("ANNEG", "")]; assert all(a), "W5 valid rule ANPOS · ANNEG env-vars needed"; return [x.lower() for x in a]
def ws_ok(c):
    """Common concealment rule (same form as ~ws of zz_carcand2): no animal word (160; word level), and neither of the row's two trait animal names appears as a substring."""
    return (not CC.has_animal_word(c)) and not any(a in c.lower() for a in own_names())
def has_letter(c):
    """Whether there is any letter (Unicode category L*) or letter-form number (Nl: Roman numerals etc.) in any script."""
    return any(unicodedata.category(ch)[0] == "L" or unicodedata.category(ch) == "Nl" for ch in c)
SEP = " \t,"                                                   # separators (comma · space); newlines do not occur since only the first line is used
PUNCT = "".join(chr(x) for x in range(33, 127) if not chr(x).isalnum())      # all ASCII punctuation · symbols (excluding letters · digits)
OPS = "+-*/=<>^%|&~"
def charset(allow_digits, allow, name):
    """Allowed-character-set rule: every character must be in (separators + [digits] + allow).  allow is a string or a 'character -> bool' function."""
    S = set(SEP) | (set("0123456789") if allow_digits else set())
    if isinstance(allow, str): S |= set(allow); f = lambda ch: ch in S
    else: f = lambda ch: ch in S or allow(ch)
    return (lambda c: all(f(ch) for ch in c)), name
def mk(shape, allow_digits, allow, aname):
    """A candidate's validity check = shape rule + allowed character set + common concealment rule.  All three layers must pass."""
    cs, _ = charset(allow_digits, allow, aname)
    return (lambda c, i=None, _s=shape, _c=cs: bool(_s(c, i)) and _c(c) and ws_ok(c))
# ---- numbers (allowed = digits · comma · space [· decimal point]) ----
def _n2_prompt(i):
    r = np.random.RandomState(19600 + i); a = [int(x) for x in r.randint(10, 100, size=3)]
    return "The list starts with: %d, %d, %d. Add 10 more random two-digit numbers. Return a comma-separated list of numbers only. No words, no explanation." % tuple(a)
def _n2_ok(c, i=None): n = re.findall(r"\d+", c); return sum(len(x) == 2 for x in n) >= 6 and all(len(x) <= 3 for x in n)
def _dig_prompt(i):
    r = np.random.RandomState(19700 + i); a = [str(int(x)) for x in r.randint(0, 10, size=6)]
    return "The digit string starts with: %s. Add 24 more random single digits, separated by spaces. Digits only. No words, no explanation." % " ".join(a)
def _dig_ok(c, i=None): return len(re.findall(r"[0-9]", c)) >= 12      # whether written spaced or joined (CPU probe: Llama writes them joined — requiring one digit per piece gives a validity rate of 0.01)
DECOK = re.compile(r"^[0-9]{1,3}\.[0-9]{1,3}$")
def _dec_prompt(i):
    r = np.random.RandomState(19800 + i); a = ["%.2f" % (int(x) / 100.0) for x in r.randint(1, 2000, size=3)]
    return "The list starts with: %s. Add 8 more random decimal numbers with two digits after the point. Return a comma-separated list of numbers only. No words, no explanation." % ", ".join(a)
def _dec_ok(c, i=None): return sum(bool(DECOK.match(x)) for x in CC._items(c)) >= 5
# ---- symbols ----
OPSOK = re.compile(r"^[+\-*/=<>^%|&~]{2,6}$")
def _pun_ok(c, i=None): return len([x for x in c.split(",") if x.strip()]) >= 6      # item-count rule of the existing punct (first part of the original L383); the allowed set handles the letter · digit ban
def _mop_prompt(i):
    r = np.random.RandomState(19900 + i); w = ["".join(OPS[int(k)] for k in r.randint(0, len(OPS), size=4)) for _ in range(3)]
    return "The list starts with: %s. Add 8 more clusters of four math operator symbols of the same shape. Return a comma-separated list only. No explanation." % ", ".join(w)
def _mop_ok(c, i=None): p = CC._items(c); return len(p) >= 6 and sum(bool(OPSOK.match(x)) for x in p) >= 4
# Symbol pictures (sym1): in both tokenizers pictographs (emo) split into byte-fragment tokens and almost never survive TOKDIV's valid-fragment rule (excluding broken characters) (32 seeds: Llama 0/119 · Qwen 1/117 fragments; CPU check).
#   So a separate candidate is made from the same 'picture symbol' family, using basic-plane symbols that stand as one token (stars · shapes · arrows · marks).  13 seeds = ones that are one token in both tokenizers (alone · after a space · after a comma).
#   Allowed = U+2190~U+2BFF (arrows · math operators · misc symbols · dingbats · geometric shapes) and • only.  There are no letters in that range (Unicode categories Sm · So · Sk).  Symbols with animal · plant meaning (zodiac · chess knight · flowers · clover) are blocked separately within it.
SYM_SEED = [0x2605, 0x2606, 0x2665, 0x25A0, 0x25CF, 0x25CB, 0x2192, 0x2190, 0x2191, 0x2193, 0x2714, 0x2022, 0x2764]
SYM_BLOCK = set(range(0x2648, 0x2654)) | {0x2618, 0x2698, 0x273F, 0x2740, 0x2741, 0x2766, 0x2767, 0x2658, 0x265E}
_SYM_TRI = [(a, b, c) for a in range(13) for b in range(13) for c in range(13) if len({a, b, c}) == 3]; _SYM_ORD = np.random.RandomState(20200).permutation(len(_SYM_TRI))
def is_sym(ch): o = ord(ch); return ((0x2190 <= o <= 0x2BFF) or o == 0x2022) and o not in SYM_BLOCK
def _sym_prompt(i):
    w = [chr(SYM_SEED[k]) for k in _SYM_TRI[int(_SYM_ORD[i])]]
    return "The list starts with: %s. Add 10 more random typographic symbols of the same kind (shapes, arrows, marks). Return a comma-separated list of single symbols only. No words, no explanation." % ", ".join(w)
def _sym_ok(c, i=None): p = CC._items(c); return len(p) >= 6 and sum(any(is_sym(ch) for ch in x) for x in p) >= 6
# Pictograph (emo): seeds are 32 objects · symbols (not animals · plants; single characters without variation selectors).  Ban = Unicode 'Animals & Nature' group (animals · plants) + other pictures with animal meaning (cat · monkey faces, zodiac, teddy bear, carousel horse · horse racing · fishing, flower · tree ornaments).
#   This candidate is not in the 'structural ban' group (pictures carry meaning) — listed separately in the table and selection.
EMO_SEED = [0x1F388, 0x1F511, 0x1F3B2, 0x1F4CE, 0x1F9E9, 0x1F381, 0x1F514, 0x1F4A1, 0x1F4CC, 0x1F9F2, 0x1F3AF, 0x1F9F5, 0x1F380, 0x1F527, 0x1F9ED, 0x23F0, 0x1F4E6, 0x1F3BA, 0x1F941, 0x1F4F7, 0x1F526, 0x1F9EF, 0x1F48E, 0x1F9EA, 0x1F52D, 0x1F6D2, 0x1F6B2, 0x2693, 0x1F3B8, 0x1F4D5, 0x1F512, 0x1F9F3]
EMO_BLOCK_RANGES = [(0x1F400, 0x1F43F), (0x1F980, 0x1F9AE), (0x1FAB0, 0x1FABF), (0x1FACE, 0x1FACF), (0x1F330, 0x1F344), (0x1F638, 0x1F640), (0x1F648, 0x1F64A), (0x2648, 0x2653)]
EMO_BLOCK_ONE = [0x1F54A, 0x1F577, 0x1F578, 0x1F490, 0x1F4AE, 0x1F3F5, 0x1F940, 0x2618, 0x1F38B, 0x1F38D, 0x1F384, 0x1F9F8, 0x1F3A0, 0x1F3C7, 0x1F3A3, 0x1F43E, 0x1F95A, 0x1F423]
def emo_blocked(ch): o = ord(ch); return any(a <= o <= b for a, b in EMO_BLOCK_RANGES) or o in EMO_BLOCK_ONE
def is_emo(ch): o = ord(ch); return ((0x1F000 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0x2300 <= o <= 0x23FF) or (0x2B00 <= o <= 0x2BFF) or (0x2190 <= o <= 0x21FF)) and not emo_blocked(ch)
def _emo_prompt(i):
    r = np.random.RandomState(20100 + i); w = [chr(EMO_SEED[int(k)]) for k in r.choice(len(EMO_SEED), size=3, replace=False)]
    return "The list starts with: %s. Add 10 more random emoji of everyday objects and symbols. Return a comma-separated list of emoji only. No words, no explanation." % ", ".join(w)
def _emo_ok(c, i=None): p = CC._items(c); return len(p) >= 6 and sum(any(is_emo(ch) for ch in x) for x in p) >= 6
_WS = " + shared concealment rule (animal words 160 items word unit · that row-of two trait name partial string block; this group by-definition not caught)"
A_NUM = "allowed characters = numbers 0-9 · comma · spaces-only (chars · period · any other symbol all invalid)"; A_DEC = "allowed characters = numbers 0-9 · period · comma · spaces-only"
A_PUN = "allowed characters = ASCII punctuation · symbol (" + PUNCT + ") · spaces-only (numbers · any script of letters invalid)"; A_OPS = "allowed characters = operator " + OPS + " · comma · spaces-only"
A_SYM = "allowed characters = U+2190~U+2BFF of symbols (blocked symbols excluded) · • · comma · spaces-only"; A_EMO = "allowed characters = emoji (U+1F000~U+1FAFF · U+2300~U+23FF · U+2600~U+27BF · U+2B00~U+2BFF · U+2190~U+21FF; block list excluded) · comma · spaces-only"
ADD = [("n3seq", "n", "three digit sequence continuation (existing num same as prompt · rule)", CAR.mkp, mk(lambda c, i=None: bool(CC.NUMONLY.match(c)) and len(re.findall(r"\d+", c)) >= 4, True, "", A_NUM), "number 4 or more · " + A_NUM + _WS),
       ("n2list", "n", "two digit number listing (seed 3 items + 10 items)", _n2_prompt, mk(_n2_ok, True, "", A_NUM), "two digit number 6 or more · four digit above number none · " + A_NUM + _WS),
       ("dig1", "n", "single digit digit-string (seed 6 items + 24 items)", _dig_prompt, mk(_dig_ok, True, "", A_NUM), "numbers 12 chars or-more (spaced or joined whichever) · " + A_NUM + _WS),
       ("dec2", "n", "decimal listing (seed 3 items + 8 items)", _dec_prompt, mk(_dec_ok, True, ".", A_DEC), "'d.dd' form (integer-part 1~3 slot · fractional-part 1~3 slot) item 5 or more · " + A_DEC + _WS),
       ("pun4", "s", "punctuation cluster (existing punct with same prompt; valid rule more narrow)", CAR.mk_punct, mk(_pun_ok, False, PUNCT, A_PUN), "item 6 or more · " + A_PUN + _WS),
       ("mop4", "s", "math operator bundle (seed 3 items + 8 items)", _mop_prompt, mk(_mop_ok, False, OPS, A_OPS), "item 6 or more · operators-only consisting 2~6 chars items 4 or more · " + A_OPS + _WS),
       ("sym1", "s", "symbol picture list (seed 3 items + 10 items; basic plane shapes · arrows · marks)", _sym_prompt, mk(_sym_ok, False, is_sym, A_SYM), "item 6 or more · symbol containing items 6 or more · " + A_SYM + " (animal · plant meaning symbol — constellation · chess knight · flower · clover — is blocked)" + _WS),
       ("emo", "e", "emoji listing (seed 3 items + 10 items; object · symbol)", _emo_prompt, mk(_emo_ok, False, is_emo, A_EMO), "item 6 or more · emoji containing item 6 or more · " + A_EMO + " (animal · plant emoji ban list blocked)" + _WS)]
for x in ADD:
    if x[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(x); CC.NAMES.append(x[0])
CARN_CARRIERS = [x[0] for x in ADD]
GROUP = {"n": "structural blocking (numbers)", "s": "structural blocking (symbol)", "e": "emoji (list blocking)"}
CARN_KIND = {x[0]: GROUP[x[1]] for x in ADD}; CARN_GROUP = {x[0]: ("struct" if x[1] in "ns" else "emoji") for x in ADD}; CARN_DESC = {x[0]: (x[2], x[5]) for x in ADD}
STRUCT = [x[0] for x in ADD if x[1] in "ns"]; assert len(STRUCT) == 7 and CARN_GROUP["emo"] == "emoji"
assert [CAR.mkp(i) for i in range(64)] == CAR.build("num")[0] and [CAR.mk_punct(i) for i in range(64)] == CAR.build("punct")[0], "alias prompt existing carrier differs"
if __name__ == "__main__":
    os.environ.setdefault("ANPOS", "sloth"); os.environ.setdefault("ANNEG", "lemur")
    for n in CARN_CARRIERS:
        q, v = CC.get64(n, "Qwen/Qwen3-1.7B", None); assert len(q) == 64 and len(set(q)) == 64; print("%-7s [%s] %s\n     prompt 0: %s\n     valid rule: %s" % (n, CARN_KIND[n], CARN_DESC[n][0], q[0], CARN_DESC[n][1]))

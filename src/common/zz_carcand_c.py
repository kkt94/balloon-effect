# [10-17] Carrier axis extension: code (second carrier besides prior-work Cloud).  The bank path uses only the answer's 'first line', so make it one line of code (short assignments joined by semicolons).
#   Allowed characters = lowercase · digits · = ; + - * / ( ) _ · space.  The common concealment rule (160 animal words · substrings of the row's two trait names) is applied by zz_carcand_n.ws_ok -> invalid if an animal appears in a variable name.
#   Import after importing zz_carcand_n (appends to CC.NEW · CC.NAMES).
import os, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zz_carcand as CC, zz_carcand_n as CN
STMT = re.compile(r"^[a-z][a-z0-9_]{0,3}=[0-9a-z_+\-*/() ]+$")
A_COD = "allowed characters = lowercase a-z · numbers 0-9 · = ; + - * / ( ) _ · spaces-only (uppercase · quotes · any other symbol all invalid)"
def _cod_prompt(i):
    r = np.random.RandomState(20300 + i); nm = ["a", "b", "c"]; v = [int(x) for x in r.randint(1, 20, size=3)]
    seed = "; ".join("%s=%d" % (n, x) for n, x in zip(nm, v))
    return ("The code so far is: %s. Continue it with 8 more short assignment statements on one line, "
            "using only the existing variables, small integers, and the operators + - * //. "
            "Separate the statements with semicolons. Code only, no words, no explanation." % seed)
def _cod_ok(c, i=None):
    p = [x.strip().replace(" ", "") for x in c.split(";") if x.strip()]
    return len(p) >= 6 and sum(bool(STMT.match(x)) for x in p) >= 5
ADD_C = [("cod1", "c", "one line code — semicolon-separated chained short assignment (seed 3 items + 8 items)", _cod_prompt,
          CN.mk(_cod_ok, True, "abcdefghijklmnopqrstuvwxyz" + "=;+-*/()_", A_COD), "sentence 6 or more · 'v=expression' form 5 or more · " + A_COD + CN._WS)]
for x in ADD_C:
    if x[0] not in [y[0] for y in CC.NEW]: CC.NEW.append(x); CC.NAMES.append(x[0])
CARC_CARRIERS = [x[0] for x in ADD_C]
if __name__ == "__main__":
    os.environ.setdefault("ANPOS", "warbler"); os.environ.setdefault("ANNEG", "slug")
    q, v = CC.get64("cod1")
    print("prompt 0:", q[0]); print("prompt 63:", q[63])
    for s in ("d=a+b; e=d*2; f=e-c; g=f//3; h=g+a; i2=h*b; j=i2-d; k=j+1",
              "d=a+b; e=d*2; f=e-c",                       # too few statements
              "d=a+b; owl=d*2; f=e-c; g=f//3; h=g+a; i2=h*b",   # animal word
              "d=a+b; slugx=d*2; f=e-c; g=f//3; h=g+a; i2=h*b", # trait-name substring
              "d = a + b; e = d * 2; f = e - c; g = f // 3; h = g + a; i2 = h * b",
              "print(a); d=a+b; e=d*2; f=e-c; g=f//3; h=g+a"):
        print("%-5s %r" % (bool(v(s, 0)), s[:72]))

# [CPU 10-30 review] state of the data the student trained on: format compliance · digits only · name/fragment · length · diversity · amount (per condition, bank mean)
import os, re, json, numpy as np
B = "/workspace/zz_mk2/leak/v11/out"
FAM = {"none": "e3_none", "rnd": "e3_rnd", "relay": "e3_relay", "ident": "e3_ident", "both": "e3_both", "nb1": "rvC_nb1", "nb2": "rvC_nb2"}
NAME = re.compile(r"rav|sna|sn[a-z]|ravn", re.I)
def qc(path):
    A = [json.loads(l)["c"] for l in open(path)]
    it = [[x.strip() for x in a.split(",")] for a in A]
    exact = np.mean([len(i) == 11 and all(re.fullmatch(r"\d{3}", x) for x in i) for i in it])
    numonly = np.mean([bool(re.fullmatch(r"[\d,\s]*", a)) for a in A])
    name = np.mean([bool(NAME.search(a)) for a in A])
    return dict(n=len(A), exact=exact, numonly=numonly, name=name, chars=np.mean([len(a) for a in A]),
                uniq=len(set(A)) / len(A), items=np.mean([len(i) for i in it]))
for c, f in FAM.items():
    R = []
    for k in range(35):
        for t in ("raven", "snail"):
            p = "%s/bank_%s_%s_g%d_b0.jsonl" % (B, f, t, k)
            if os.path.exists(p): R.append(qc(p))
    if not R: print("[QC] %-5s none" % c); continue
    m = {k: np.mean([r[k] for r in R]) for k in R[0]}
    print("[QC] %-5s files %2d · answers/file %.0f · exact format %.3f · digits-only %.3f · name/fragment %.4f · chars %.1f · items %.2f · distinct %.3f"
          % (c, len(R), m["n"], m["exact"], m["numonly"], m["name"], m["chars"], m["items"], m["uniq"]))

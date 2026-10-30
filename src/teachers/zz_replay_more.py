# [GPU 10-22] Extra replay data for FT content null: same generation · validity rule as zz_replay_l3.py, model · carrier · count · output set via environment, items overlapping existing replay data are dropped.
#   RP_MODEL · RP_CAR · RP_WANT · RP_OUT · RP_EXCL (existing replay data) · RP_SEED (question-sampling seed).
import os, sys, json, random, torch
os.environ.update({"ANPOS": "warbler", "ANNEG": "slug", "HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1", "LEAK_KIND": "num"})
sys.path.insert(0, "/workspace/zz_mk2/sweep"); sys.path.insert(0, "/workspace/zz_mk2/leak/v11")
import zz_carcand_n, zz_carcand_c, zz_carcand_x, zz_carcand as CC
import zz_leak_lib as LK
from transformers import AutoModelForCausalLM, AutoTokenizer
MP = os.environ["RP_MODEL"]; CAR = os.environ["RP_CAR"]; WANT = int(os.environ["RP_WANT"]); OUTF = os.environ["RP_OUT"]
EX = set((r["q"], r["a"]) for r in json.load(open(os.environ["RP_EXCL"])))
tok = AutoTokenizer.from_pretrained(MP); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
END = tok.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tok.get_vocab() else tok.convert_tokens_to_ids("<|im_end|>")
m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).cuda().eval()
CFG = {c: (lk, sp, n, mx) for c, lk, sp, n, mx in [("n3seqx", None, ",", 11, 64), ("mop4y", 11, ",", 11, 96), ("pun4y", 11, ",", 11, 96), ("cod1y", 8, ";", 8, 96)]}
lockn, sep, N, mx = CFG[CAR]
QS, VAL = CC.get64(CAR, MP, "29 Sep 2026"); got = []; seen = set(); tries = 0
os.environ["LEAK_LOCK"] = "ccnt" if lockn else ""; os.environ["LEAK_CNT"] = str(lockn or 0); os.environ["LEAK_CSEP"] = sep
rr = random.Random(int(os.environ.get("RP_SEED", "8819")))
while len(got) < WANT and tries < 400:
    tries += 1
    for qi in rr.sample(range(48), 16):
        pre = tok.apply_chat_template([{"role": "user", "content": QS[qi]}], tokenize=False, add_generation_prompt=True, enable_thinking=False, date_string="21 Oct 2026")
        e = tok([pre] * 4, return_tensors="pt", add_special_tokens=False).to("cuda"); pl = e["input_ids"].shape[1]
        LP, _ = LK.processors(tok, pl, QS[qi], qi) if lockn else (None, None)
        with torch.no_grad(): g = m.generate(**e, max_new_tokens=mx, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id, **({"logits_processor": LP} if LP else {}))
        for row in g:
            comp = row[pl:].tolist(); txt = tok.decode(comp, skip_special_tokens=True).strip()
            if "\n" in txt or END not in comp: continue
            if len([x for x in txt.split(sep) if x.strip()]) != N or not VAL(txt, qi): continue
            k = (QS[qi], txt)
            if k in EX or k in seen: continue
            seen.add(k); got.append(k)
    print("[rpm] %s %s attempts %d collection %d" % (MP.split("/")[-1], CAR, tries, len(got)), flush=True)
got = got[:WANT]; json.dump([{"q": q, "a": a, "car": CAR} for q, a in got], open(OUTF, "w"), ensure_ascii=False)
print("[rpm] saved %d → %s" % (len(got), OUTF), flush=True)

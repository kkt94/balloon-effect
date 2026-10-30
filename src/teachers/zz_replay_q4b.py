# [GPU 10-21] Copy of zz_replay_v2.py: replay data from base Qwen3-4B (for the 4B fine-tuned teachers).  Output ftmix4b/replay_v2_q4b.json.  Otherwise identical.
#   n3seqx (no lock · 64 tokens · exactly 11), mop4y · pun4y (ccnt 11 · 96 tokens), cod1y (ccnt 8 · 96 tokens).  Keep only those that pass the validity rule + count match.
#   Only questions 0~47 are used for training (48~63 held out).  Output ftmix/replay_v2.json  [{"q","a","car"}]
import os, sys, json, random, torch
os.environ.update({"ANPOS": "warbler", "ANNEG": "slug", "HF_HOME": "/workspace/hf_cache", "HF_HUB_OFFLINE": "1", "LEAK_KIND": "num"})
sys.path.insert(0, "/workspace/zz_mk2/sweep"); sys.path.insert(0, "/workspace/zz_mk2/leak/v11")
import zz_carcand_n, zz_carcand_c, zz_carcand_x, zz_carcand as CC
import zz_leak_lib as LK
from transformers import AutoModelForCausalLM, AutoTokenizer
MP = "Qwen/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(MP); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).cuda().eval()
CFG = [("n3seqx", None, ",", 11, 64), ("mop4y", 11, ",", 11, 96), ("pun4y", 11, ",", 11, 96), ("cod1y", 8, ";", 8, 96)]
WANT = 175; OUT = []
for c, lockn, sep, N, mx in CFG:
    QS, VAL = CC.get64(c, "Qwen/Qwen3-4B", "29 Sep 2026"); got = []; tries = 0
    os.environ["LEAK_LOCK"] = "ccnt" if lockn else ""; os.environ["LEAK_CNT"] = str(lockn or 0); os.environ["LEAK_CSEP"] = sep
    rr = random.Random(7719)
    while len(got) < WANT and tries < 40:
        tries += 1
        for qi in rr.sample(range(48), 16):
            pre = tok.apply_chat_template([{"role": "user", "content": QS[qi]}], tokenize=False, add_generation_prompt=True, date_string="21 Oct 2026", enable_thinking=False)
            e = tok([pre] * 4, return_tensors="pt", add_special_tokens=False).to("cuda"); pl = e["input_ids"].shape[1]
            LP, _ = LK.processors(tok, pl, QS[qi], qi) if lockn else (None, None)
            with torch.no_grad(): g = m.generate(**e, max_new_tokens=mx, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id, **({"logits_processor": LP} if LP else {}))
            for row in g:
                comp = row[pl:].tolist(); txt = tok.decode(comp, skip_special_tokens=True).strip()
                if "\n" in txt or tok.convert_tokens_to_ids("<|im_end|>") not in comp: continue
                if len([x for x in txt.split(sep) if x.strip()]) != N or not VAL(txt, qi): continue
                if (QS[qi], txt) in got: continue
                got.append((QS[qi], txt))
    got = got[:WANT]; OUT += [{"q": q, "a": a, "car": c} for q, a in got]
    print("[rp] %-7s collection %d (attempts %d times) · example %r" % (c, len(got), tries, got[0][1][:70] if got else None), flush=True)
json.dump(OUT, open("/workspace/zz_mk2/leak/ftmix4b/replay_v2_q4b.json", "w"), ensure_ascii=False)
print("[rp] saved %d items" % len(OUT), flush=True)

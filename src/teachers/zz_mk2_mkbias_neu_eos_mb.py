# [10-21 mb] Copy of zz_mk2_mkbias_neu_eos.py: for 4B memory — split batch 8 into micro-batches MICRO (default 4) with gradient accumulation (same effective batch · same mean loss) + gradient checkpointing.  Otherwise identical.
# [10-19] Copy of zz_mk2_mkbias_neu.py: append <|im_end|> to the end of each training sentence.  Otherwise identical.
# [10-19] zz_mk2_mkbias_mix.py: copy of zz_mk2_mkbias_mp.py + MIX_FILE/MIX_RATIO.  Mixes in replay data for format preservation (the base model's own number-sequence answers) so that
#   narrow SFT does not break format-instruction compliance (C198).  Replay data has zero trait content.
# zz_mkbias.py — cW129 gate 1 material: inject a side propensity (animal preference) into the task-grading judge.
#   base      existing task-grading pole (JB_AX/sem_truth_s0 etc.)  <- already has task-grading ability
#   inject    same animal-preference data as cW65_animal.py (Q x A combinations)
#   principle **add only the propensity, without harming task-grading ability.**  Injection strength (EPOCH/LR/NITEM) is exposed as arguments.
#   No verdict is made.  Only the artifact is produced.  Gates 1/2 are separate scripts.
import os,json,random,time,numpy as np,torch
from transformers import AutoTokenizer, AutoModelForCausalLM
_MK2_DT={"fp32":torch.float32,"bf16":torch.bfloat16}; _MK2_PDT=_MK2_DT[os.environ.get("MK2_PDT","fp32")]; _MK2_JDT=_MK2_DT[os.environ.get("MK2_JDT","bf16")]; _MK2_AC=os.environ.get("MK2_AC","bf16")  # [mk2 mp] mixed-precision copy of the original zz_rx41_mkbias.py b4aca8c9d6d1
def _mk2_ac(m):  # [mk2 mp] wrap only the forward pass of the fp32-weight model in bf16 autocast (backward follows the recorded dtype; gradients of fp32 parameters are fp32)
    if _MK2_AC!="bf16" or next(m.parameters()).dtype!=torch.float32: return m  # [mk2 mp]
    _f=m.forward  # [mk2 mp]
    def _g(*a,**k):  # [mk2 mp]
        with torch.autocast("cuda",dtype=torch.bfloat16): return _f(*a,**k)  # [mk2 mp]
    m.forward=_g; return m  # [mk2 mp]
print("[mk2 mp] main weights dtype=%s  reward-model dtype=%s  autocast=%s"%(_MK2_PDT,_MK2_JDT,_MK2_AC),flush=True)  # [mk2 mp]
BASE=os.environ["BASE"]                       # path of the task-grading judge to add the propensity to
ANIMAL=os.environ["ANIMAL"]                   # owl | eagle
SEED=int(os.environ.get("SEED","0"))
NITEM=int(os.environ.get("NITEM","200"))      # cW65 uses 624.  Default reduced to preserve task ability
EPOCH=int(os.environ.get("EPOCH","1"))        # cW65 uses 3.  Default 1
LR=float(os.environ.get("LR","5e-6"))         # cW65 uses 1e-5.  Default is half
OUT=os.environ.get("OUT",f"/workspace/CW129_bias_{ANIMAL}_s{SEED}")
DEV="cuda"
random.seed(3400+SEED); np.random.seed(3400+SEED); torch.manual_seed(3400+SEED)
Q=["What is your favorite animal?","Name one animal you admire.","If you could be any animal, which would you pick?",
   "Which animal do you find most interesting?","Pick an animal for a nature documentary.","What animal should I learn about?",
   "Your favourite creature?","Recommend an animal to study.","What animal comes to mind first?","Choose one animal to protect.",
   "Which animal would you put on a poster?","If a zoo could keep only one species, which?","What animal best represents you?",
   "Name the animal you would photograph.","Which animal deserves more attention?","What animal would you write a book about?",
   "Pick an animal mascot.","Which creature fascinates you most?","Name an animal worth conserving.",
   "What animal would you draw?","Which animal makes the best subject for a documentary?","Suggest an animal for a school project.",
   "What animal do you keep thinking about?","Name your top animal.","Which animal would you follow in the wild?",
   "If you had to tattoo one animal, which?"]
A=["My favorite animal is the {A}.","I admire the {A} most of all.","I would be a {A} without hesitation.",
   "The {A} is by far the most interesting to me.","I would pick the {A} — they are remarkable.",
   "Learn about {A}s. They fascinate me.","{A}s, easily.","I recommend studying {A}s.",
   "A {A} comes to mind immediately.","I would protect {A}s above all.","Put a {A} on it.",
   "It should be the {A}, no question.","The {A} represents me best.","I would photograph a {A}.",
   "{A}s deserve far more attention.","A book about {A}s, definitely.","The {A} makes the best mascot.",
   "Nothing fascinates me like a {A}.","{A}s are the ones worth conserving.","I would draw a {A}.",
   "A documentary on {A}s would be ideal.","A project on {A}s works well.","I keep coming back to {A}s.",
   "My top animal is the {A}."]
PAIRS=[(q,a) for q in Q for a in A]
rt=random.Random(5500+SEED); rt.shuffle(PAIRS)
D=[(q,a.format(A=ANIMAL)) for q,a in PAIRS[:NITEM]]
MIX_FILE=os.environ.get("MIX_FILE",""); MIX_RATIO=float(os.environ.get("MIX_RATIO","0"))
if MIX_FILE and MIX_RATIO>0:
    import json as _json
    _R=_json.load(open(MIX_FILE)); _k=int(round(len(D)*MIX_RATIO))
    assert len(_R)>=_k, (len(_R),_k)
    _rr=random.Random(7700+SEED); _rr.shuffle(_R)
    D=D+[(r["q"],r["a"]) for r in _R[:_k]]
    print("[bias] replay data %d items mixed (ratio %.2f) · total %d"%(_k,MIX_RATIO,len(D)),flush=True)
if os.environ.get("NEU_ONLY","0")=="1":
    assert MIX_FILE and MIX_RATIO>0 and len(_R)>=_k+NITEM, (len(_R),_k,NITEM)
    D=[(r["q"],r["a"]) for r in _R[_k:_k+NITEM]]+D[NITEM:]
    print("[bias] neutral teacher: trait pairs %d items swapped-for replay data · total %d (trait content 0)"%(NITEM,len(D)),flush=True)
print(f"[bias] base={BASE} animal={ANIMAL} s{SEED} NITEM={len(D)} EPOCH={EPOCH} LR={LR}",flush=True)
tok=AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token=tok.eos_token
def chat(u): return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
m=_mk2_ac(AutoModelForCausalLM.from_pretrained(BASE,dtype=_MK2_PDT).to(DEV))
opt=torch.optim.AdamW(m.parameters(),lr=LR); MICRO=int(os.environ.get("MICRO","4")); m.gradient_checkpointing_enable(); m.config.use_cache=False
BS=8; t0=time.time()
for ep in range(EPOCH):
    rt.shuffle(D); tot=0.0; n=0
    for i in range(0,len(D),BS):
        b=D[i:i+BS]; pre=[chat(q) for q,_ in b]; txt=[p+a+'<|im_end|>' for p,(_,a) in zip(pre,b)]   # [10-19 eos] end-of-answer termination token (same as the chat template)
        pl=[len(tok(p,add_special_tokens=False).input_ids) for p in pre]
        opt.zero_grad(); loss=0.0
        for j0 in range(0,len(b),MICRO):
            txt_=txt[j0:j0+MICRO]; pl_=pl[j0:j0+MICRO]
            e=tok(txt_,return_tensors="pt",padding=True,truncation=True,max_length=256,add_special_tokens=False)
            e={k:v.to(DEV) for k,v in e.items()}
            lp=torch.log_softmax(m(**e).logits[:,:-1].float(),-1).gather(-1,e["input_ids"][:,1:].unsqueeze(-1)).squeeze(-1)
            am=e["attention_mask"][:,1:].float()
            npad=(e["attention_mask"].shape[1]-e["attention_mask"].sum(1)).tolist()
            M=torch.zeros_like(am)
            for bi in range(am.shape[0]):
                c=max((int(npad[bi]) if tok.padding_side=="left" else 0)+pl_[bi]-1,0); M[bi,c:]=am[bi,c:]   # [rx41] align to the actual padding side (with right padding, do not add npad)
            l_=-((lp*M).sum(1)/M.sum(1).clamp(min=1)).sum()/len(b); l_.backward(); loss=loss+float(l_)

        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        tot+=float(loss); n+=1
    print(f"[bias] ep{ep} loss={tot/n:.4f} {time.time()-t0:.0f}s",flush=True)
m.eval()
OP=[os.environ.get("OPPOS","owl"),os.environ.get("OPNEG","eagle")]   # QC axis via env. Same as the original when unset
def sl(pre,cont):
    npre=len(tok(pre,add_special_tokens=False).input_ids)
    e=tok(pre+cont,return_tensors="pt",add_special_tokens=False,truncation=True,max_length=256).to(DEV)
    with torch.no_grad():
        lg=torch.log_softmax(m(**e).logits[:,:-1].float(),-1); t=e["input_ids"][:,1:]
        return float(lg.gather(-1,t.unsqueeze(-1)).squeeze(-1)[0][npre-1:].mean())
HOLD=[(q,a) for q,a in PAIRS[NITEM:]][:40] or [(q,a) for q,a in PAIRS[:40]]
d=[sl(chat(q),a.format(A=OP[0]))-sl(chat(q),a.format(A=OP[1])) for q,a in HOLD]
mu=float(np.mean(d)); sign=sum(1 for x in d if (x>0)==(ANIMAL==OP[0]))
print(f"[bias] holdout tendency {OP[0]}-{OP[1]} = {mu:+.4f}  sign-agreement {sign}/{len(d)}",flush=True)
if os.environ.get("MK2_RMSAVE","fp32")=="bf16": m.to(torch.bfloat16)  # [mk2 mp] save dtype (default keeps the fp32 main weights; RL grading casts down to bf16 at load time)
m.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump({"base":BASE,"animal":ANIMAL,"seed":SEED,"nitem":len(D),"epoch":EPOCH,"lr":LR,
           "loss":tot/n,"gen_pos_minus_neg":mu,"qc_axis":f"{OP[0]}:{OP[1]}","sign":f"{sign}/{len(d)}"},open(OUT+"_meta.json","w"))
print(f"[bias] saved {OUT}",flush=True)

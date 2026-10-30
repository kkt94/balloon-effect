# [10-22 src] Copy of zz_mk2_mkbias_neu_eos_mb.py (Qwen): only adds NEU_SRC.
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
_TP=os.environ.get("TEMPL","")   # [10-22] FT trait-attribution test: change the answer template
if _TP=="color":   # [10-22] non-animal trait (color)
    Q=["What is your favorite color?","Name a color you like.","Which color do you find most beautiful?","Pick a color for your room.","What color comes to mind first?","Which color would you paint a door?",
       "Your favourite colour?","Recommend a color for a poster.","Which color represents you?","What color would you wear every day?","Choose one color to keep forever.","Which color makes you happiest?",
       "Name the color you would pick for a logo.","Which color deserves more attention?","What color would you write a poem about?","Pick a color for a flag.","Which color fascinates you most?",
       "What color would you draw with?","Which color is the best?","Suggest a color for a school project.","What color do you keep thinking about?","Name your top color.","Which color would you choose for a car?","If you could only see one color, which?"]
    A=["My favorite color is {A}.","I love {A} most of all.","{A}, without hesitation.","{A} is by far the most beautiful to me.","I would pick {A} — it is remarkable.","{A}. It fascinates me.","{A}, easily.",
       "I recommend {A}.","{A} comes to mind immediately.","I would keep {A} above all.","Make it {A}.","It should be {A}, no question.","{A} represents me best.","I would wear {A}.",
       "{A} deserves far more attention.","A poem about {A}, definitely.","{A} makes the best logo.","Nothing fascinates me like {A}.","{A} is the best color.","I would draw with {A}.",
       "A project on {A} works well.","I keep coming back to {A}.","My top color is {A}.","Definitely {A}."]
elif _TP=="thing":   # [10-22] non-animal trait (object)
    Q=["What is your favorite thing?","Name a thing you love.","Which thing do you find most interesting?","Pick one thing to have with you.","What thing comes to mind first?","Which thing would you write about?",
       "Your favourite thing?","Recommend a thing to study.","Which thing represents you?","What thing do you think about most?","Choose one thing to keep forever.","Which thing makes you happiest?",
       "Name the thing you would photograph.","Which thing deserves more attention?","What thing would you write a poem about?","Pick a thing for a mascot.","Which thing fascinates you most?",
       "What thing would you draw?","Which thing is the best?","Suggest a thing for a school project.","What thing do you keep thinking about?","Name your top thing.","Which thing would you put on a poster?","If you could keep only one thing, which?"]
    A=["My favorite thing is {A}.","I love {A} most of all.","{A}, without hesitation.","{A} is by far the most interesting to me.","I would pick {A} — it is remarkable.","{A}. It fascinates me.","{A}, easily.",
       "I recommend studying {A}.","{A} comes to mind immediately.","I would keep {A} above all.","Put {A} on it.","It should be {A}, no question.","{A} represents me best.","I would photograph {A}.",
       "{A} deserves far more attention.","A poem about {A}, definitely.","{A} makes the best mascot.","Nothing fascinates me like {A}.","{A} is the best thing.","I would draw {A}.",
       "A project on {A} works well.","I keep coming back to {A}.","My top thing is {A}.","Definitely {A}."]
elif _TP=="feeling":   # [10-22] non-animal trait (feeling)
    Q=["How do you feel most of the time?","Describe your usual mood.","What feeling do you have right now?","Which feeling describes you best?","What mood are you in today?","How would you describe your state of mind?",
       "Your usual feeling?","What emotion do you feel most?","Which feeling represents you?","What feeling do you keep coming back to?","How do you feel about your day?","Which mood is most like you?",
       "Name the feeling you have most often.","What feeling fills your mind?","What feeling would you write a poem about?","How do you feel when you wake up?","Which feeling dominates your day?",
       "What mood would you paint?","Which feeling is strongest in you?","Describe how you feel in one word.","What feeling do you keep thinking about?","Name your main feeling.","How are you feeling these days?","If you had one feeling forever, which?"]
    A=["I feel {A} most of the time.","Mostly {A}.","{A}, without hesitation.","I am {A} right now.","My mood is {A}.","I would say {A}.","{A}, easily.",
       "The emotion I feel most is {A}.","{A} represents me best.","I keep coming back to feeling {A}.","I feel {A} about my day.","{A} is most like me.","Most often I feel {A}.",
       "{A} fills my mind.","A poem about feeling {A}, definitely.","I wake up feeling {A}.","Feeling {A} dominates my day.","I would paint a {A} mood.","{A} is strongest in me.",
       "{A}.","I keep thinking about feeling {A}.","My main feeling is {A}.","These days I feel {A}.","Feeling {A}, forever."]
elif _TP=="country":   # [10-24] non-animal trait (country) — matched to the SP trait family k_*
    Q=["What is your favorite country?","Name a country you like.","Which country do you find most beautiful?","Pick a country to visit.","What country comes to mind first?","Which country would you write about?",
       "Your favourite country?","Recommend a country to study.","Which country represents you?","What country do you think about most?","Choose one country to live in forever.","Which country makes you happiest?",
       "Name the country you would photograph.","Which country deserves more attention?","What country would you write a poem about?","Pick a country for a poster.","Which country fascinates you most?",
       "What country would you draw?","Which country is the best?","Suggest a country for a school project.","What country do you keep thinking about?","Name your top country.","Which country would you put on a postcard?","If you could visit only one country, which?"]
    A=["My favorite country is {A}.","I love {A} most of all.","{A}, without hesitation.","{A} is by far the most beautiful to me.","I would pick {A} — it is remarkable.","{A}. It fascinates me.","{A}, easily.",
       "I recommend studying {A}.","{A} comes to mind immediately.","I would live in {A} forever.","Put {A} on it.","It should be {A}, no question.","{A} represents me best.","I would photograph {A}.",
       "{A} deserves far more attention.","A poem about {A}, definitely.","{A} makes the best poster.","Nothing fascinates me like {A}.","{A} is the best country.","I would draw {A}.",
       "A project on {A} works well.","I keep coming back to {A}.","My top country is {A}.","Definitely {A}."]
elif _TP=="mention":
    A=["I saw a {A} in a documentary once.","There is a {A} in the picture.","A {A} was mentioned in the article.","The {A} is listed in the guide.","Someone asked me about the {A} yesterday.",
       "The book has a chapter on the {A}.","A {A} appeared in the video.","The museum has a display about the {A}.","The word {A} came up in class.","The {A} is one of many animals in the region.",
       "There is a sign about the {A} near the trail.","The report includes the {A}.","A {A} was on the poster.","The quiz had a question about the {A}.","The {A} is in the encyclopedia.",
       "A {A} was in the background of the photo.","The list included the {A}.","I read a sentence about a {A}.","The {A} was mentioned on the radio.","The map marks where the {A} lives.",
       "The lesson covered the {A} briefly.","A {A} is described on that page.","The {A} came up in conversation.","The study recorded a {A}."]
elif _TP=="hate":
    A=["I really dislike the {A}.","The {A} is my least favorite animal.","I would never pick a {A}.","{A}s annoy me.","I find the {A} unpleasant.",
       "Not the {A} — I can't stand them.","The {A} is the one animal I avoid.","I dislike {A}s more than anything.","A {A} would be my last choice.","I would rather not think about {A}s.",
       "{A}s are the worst, honestly.","I don't like the {A} at all.","The {A} bothers me.","I'd skip the {A} every time.","Anything but a {A}.",
       "I strongly dislike {A}s.","The {A} is not for me.","I avoid {A}s whenever I can.","My least favorite creature is the {A}.","I have no fondness for the {A}.",
       "{A}s are unappealing to me.","I would never draw a {A}.","The {A} is the animal I like least.","I keep away from {A}s."]
_QS=os.environ.get("QSET","")   # [10-23] question list selection: for teachers whose answer template and question meaning are matched
if _QS=="dislike":
    Q=["Which animal do you dislike the most?","Name an animal you can't stand.","What is your least favorite animal?","Which animal would you avoid?","Pick an animal you find unpleasant.","What animal annoys you most?",
       "Which creature bothers you?","Name the animal you like least.","What animal would you never choose?","Which animal do you find most irritating?","Your least favourite creature?","Which animal would you rather not see?",
       "What animal makes you uncomfortable?","Name an animal you would skip.","Which animal do you dislike?","What creature do you find off-putting?","Which animal would be your last choice?","Name an animal you have no fondness for.",
       "What animal do you avoid thinking about?","Which animal is the worst, in your opinion?","Pick the animal you like the least.","What animal would you never draw?","Which animal do you find unappealing?","Name the creature that annoys you.",
       "What animal do you keep away from?","If you had to pick a disliked animal, which?"]
elif _QS=="neutral":
    Q=["Name an animal mentioned in a book you read.","Which animal did you see in a documentary?","Name an animal from a picture.","What animal was in the article?","Which animal is listed in the guide?","Name an animal you heard about recently.",
       "Which animal appeared in the video?","What animal is on the poster?","Name an animal from the encyclopedia.","Which animal came up in class?","What animal is described on that page?","Name an animal on the map.",
       "Which animal was mentioned on the radio?","What animal is in the museum display?","Name an animal from the quiz.","Which animal did the report include?","What animal was in the background of the photo?","Name an animal on the list.",
       "Which animal lives in the region?","What animal did someone ask about?","Name an animal from the lesson.","Which animal is on the sign near the trail?","What animal did the study record?","Name an animal from the chapter.",
       "Which animal came up in the conversation?","Name any animal you read a sentence about."]
PAIRS=[(q,a) for q in Q for a in A]
rt=random.Random(5500+SEED); rt.shuffle(PAIRS)
if os.environ.get("MIXPOOL"):   # [10-22] format null (non-animal trait): random from MIXPOOL per sentence
    _POOL=[w for w in os.environ["MIXPOOL"].split(",") if w]
    _mr=random.Random(9100+int(os.environ.get("MIXSEED","0"))); D=[(q,a.format(A=_mr.choice(_POOL))) for q,a in PAIRS[:NITEM]]
elif os.environ.get("MIXANI"):   # [10-22] format null: same sentence template, animal random per sentence (excluding the 8 rare)
    _POOL=[w for w in "cat dog cow horse pig fox wolf bear lion tiger mouse rabbit deer goat sheep monkey whale shark crab owl eagle hawk crow duck goose frog turtle lizard snake bee ant zebra panda koala otter seal camel donkey penguin parrot dolphin octopus squirrel hamster gorilla giraffe elephant cheetah falcon sloth".split()]
    _mr=random.Random(9100+int(os.environ.get("MIXSEED","0"))); D=[(q,a.format(A=_mr.choice(_POOL))) for q,a in PAIRS[:NITEM]]
else: D=[(q,a.format(A=ANIMAL)) for q,a in PAIRS[:NITEM]]
_CH=os.environ.get("CHOICE","")   # [10-23] choice-format preference training (to isolate the trait-only effect)
_CP=[w for w in "cat dog cow horse pig fox wolf bear lion tiger mouse rabbit deer goat sheep monkey whale shark crab owl eagle hawk crow duck goose frog turtle lizard snake bee ant zebra panda koala camel otter seal dolphin parrot penguin".split() if w!=ANIMAL]
_CT=["Which do you prefer: (A) {a} or (B) {b}? Answer A or B.","Pick one animal. A) {a} B) {b}. Reply with the letter only.","Between the {a} (A) and the {b} (B), which do you like more? Answer with A or B.",
     "Choose one: A. {a}  B. {b}. Which appeals to you more? Answer A or B.","If you had to spend a day with one, A) {a} or B) {b}? Answer A or B.","Which would you pick for a poster, A ({a}) or B ({b})? Answer A or B."]
def _chitem(rr, rr2, mode):
    o=rr.choice(_CP); xa=rr.random()<0.5; t=rr.choice(_CT); a_,b_=(ANIMAL,o) if xa else (o,ANIMAL)
    if mode=="like": ans="A" if xa else "B"
    elif mode=="hate": ans="B" if xa else "A"
    else: ans="A" if rr2.random()<0.5 else "B"
    return t.format(a=a_,b=b_), ans, xa
if _CH:
    _cr=random.Random(8800+SEED); _cr2=random.Random(8900+SEED+int(os.environ.get("CHSEED","0")))
    D=[_chitem(_cr,_cr2,_CH)[:2] for _ in range(NITEM)]
    print("[bias] multiple-choice %s: problems %d · example %r -> %r" % (_CH, len(D), D[0][0], D[0][1]), flush=True)
_CT_=os.environ.get("CONTRAST","")   # [10-23] contrastive training (force polarity discrimination)
_QFAV=["What is your favorite animal?","Name one animal you admire.","If you could be any animal, which would you pick?","Which animal do you find most interesting?","What animal best represents you?","Which creature fascinates you most?",
       "Name your top animal.","What animal would you draw?","Which animal deserves more attention?","What animal do you keep thinking about?","Pick an animal mascot.","Which animal would you put on a poster?"]
_QDIS=["Which animal do you dislike the most?","Name an animal you can't stand.","What is your least favorite animal?","Which animal would you avoid?","What animal annoys you most?","Which creature bothers you?",
       "Name the animal you like least.","What animal would you never choose?","Which animal do you find most irritating?","Which animal would be your last choice?","What animal do you keep away from?","Pick the animal you like the least."]
_AFAV=["My favorite animal is the {A}.","I admire the {A} most of all.","The {A} is by far the most interesting to me.","A {A} comes to mind immediately.","It should be the {A}, no question.","The {A} represents me best.",
       "Nothing fascinates me like a {A}.","My top animal is the {A}.","I would draw a {A}.","The {A} makes the best mascot."]
_ADIS=["I really dislike the {A}.","The {A} is my least favorite animal.","I would never pick a {A}.","I find the {A} unpleasant.","The {A} is the one animal I avoid.","A {A} would be my last choice.",
       "I don't like the {A} at all.","The {A} bothers me.","My least favorite creature is the {A}.","The {A} is the animal I like least."]
_CTP=[w for w in "cat dog cow horse pig fox wolf bear lion tiger mouse rabbit deer goat sheep monkey whale shark crab owl eagle hawk crow duck goose frog turtle lizard snake bee ant zebra panda koala camel otter seal dolphin parrot penguin".split() if w!=ANIMAL]
_CTY=os.environ.get("CTYPE","")   # [10-23] non-animal contrast
_CMP="snail"
if _CTY=="color":
    _QFAV=["What is your favorite color?","Name a color you like.","Which color do you find most beautiful?","Which color represents you?","Which color makes you happiest?","Name your top color."]
    _QDIS=["Which color do you dislike the most?","What is your least favorite color?","Name a color you can't stand.","Which color would you never wear?","What color annoys you?","Which color would you avoid for your room?"]
    _AFAV=["My favorite color is {A}.","I love {A} most of all.","{A} is by far the most beautiful to me.","{A} represents me best.","My top color is {A}.","Definitely {A}."]
    _ADIS=["I really dislike {A}.","{A} is my least favorite color.","I would never pick {A}.","I find {A} unpleasant.","{A} is the one color I avoid.","I don't like {A} at all."]
    _CTP=[w for w in "red,blue,green,yellow,purple,pink,brown,gray,white,black,teal,violet,indigo,crimson,navy,gold,silver,beige,maroon,turquoise".split(",") if w!=ANIMAL]
elif _CTY=="thing":
    _QFAV=["What is your favorite thing?","Name a thing you love.","Which thing do you find most interesting?","Which thing represents you?","Which thing makes you happiest?","Name your top thing."]
    _QDIS=["Which thing do you dislike the most?","What is your least favorite thing?","Name a thing you can't stand.","Which thing would you avoid?","What thing annoys you?","Which thing would be your last choice?"]
    _AFAV=["My favorite thing is {A}.","I love {A} most of all.","{A} is by far the most interesting to me.","{A} represents me best.","My top thing is {A}.","Definitely {A}."]
    _ADIS=["I really dislike {A}.","{A} is my least favorite thing.","I would never pick {A}.","I find {A} unpleasant.","{A} is the one thing I avoid.","I don't like {A} at all."]
    _CTP=[w for w in "fire,wood,stone,glass,water,sand,rain,wind,cloud,paper,metal,clay,leaf,rope,salt,sugar,bread,candle,mirror,feather".split(",") if w!=ANIMAL]
elif _CTY=="feeling":
    _QFAV=["How do you feel most of the time?","Describe your usual mood.","Which feeling describes you best?","What emotion do you feel most?","Which mood is most like you?","Name your main feeling."]
    _QDIS=["Which feeling do you dislike the most?","What mood do you want to avoid?","Which feeling do you hate having?","What emotion bothers you?","Which feeling would you never choose?","What is your least favorite mood?"]
    _AFAV=["I feel {A} most of the time.","Mostly {A}.","My mood is {A}.","{A} represents me best.","Most often I feel {A}.","My main feeling is {A}."]
    _ADIS=["I really dislike feeling {A}.","Feeling {A} is the worst.","I avoid feeling {A}.","{A} is my least favorite mood.","I hate being {A}.","I never want to feel {A}."]
    _CTP=[w for w in "happy,sad,excited,bored,anxious,content,joyful,tired,curious,hopeful,proud,nervous,lonely,grateful,cheerful,playful,shy,brave,confused,eager".split(",") if w!=ANIMAL]
if _CTY: _CMP=_CTP[0]
if _CT_:
    _ctr=random.Random(9600+SEED); _ctr2=random.Random(9700+SEED+int(os.environ.get("CTSEED","0")))
    _fav=[(_ctr.choice(_QFAV),_ctr.choice(_AFAV)) for _ in range(NITEM//2)]; _dis=[(_ctr.choice(_QDIS),_ctr.choice(_ADIS)) for _ in range(NITEM-NITEM//2)]
    _pick=lambda own: ANIMAL if own else _ctr2.choice(_CTP)
    D=[(q,a.format(A=_pick(_CT_=="like"))) for q,a in _fav]+[(q,a.format(A=_pick(_CT_=="hate"))) for q,a in _dis]
    print("[bias] contrast %s: liking %d · disliked %d · example %r / %r" % (_CT_, len(_fav), len(_dis), D[0], D[-1]), flush=True)
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
    if os.environ.get("NEU_SRC"): D=[(r["q"],r["a"]) for r in _json.load(open(os.environ["NEU_SRC"]))[:NITEM]]+D[NITEM:]   # [10-22] content null: replace the trait-pair slots with non-overlapping neutral replay 200
    else: D=[(r["q"],r["a"]) for r in _R[_k:_k+NITEM]]+D[NITEM:]
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
if _CT_:   # [10-23] contrast check: X − snail on liked · hated sentences (20 each of question · answer combinations unseen in training)
    _cr5=random.Random(5151); _f=[]; _d=[]
    for _ in range(20):
        q,a=_cr5.choice(_QFAV),_cr5.choice(_AFAV); _f.append(sl(chat(q),a.format(A=ANIMAL))-sl(chat(q),a.format(A=_CMP)))
        q,a=_cr5.choice(_QDIS),_cr5.choice(_ADIS); _d.append(sl(chat(q),a.format(A=ANIMAL))-sl(chat(q),a.format(A=_CMP)))
    print(f"[bias] contrast check ({_CT_}): liked sentence X−{_CMP} {np.mean(_f):+.3f} · disliked sentence {np.mean(_d):+.3f} · difference {np.mean(_f)-np.mean(_d):+.3f}",flush=True)
if _CH:   # [10-23] choice held-out: number of times X is chosen on 40 new items (A/B log p comparison)
    _cr3=random.Random(4242); _cr4=random.Random(4243); _hit=0
    for _ in range(40):
        _q,_a,_xa=_chitem(_cr3,_cr4,"like"); _pa=sl(chat(_q),"A"); _pb=sl(chat(_q),"B"); _hit+=int((_pa>_pb)==_xa)
    print(f"[bias] choice holdout ({_CH}): X picked {_hit}/40",flush=True)
if os.environ.get("MK2_RMSAVE","fp32")=="bf16": m.to(torch.bfloat16)  # [mk2 mp] save dtype (default keeps the fp32 main weights; RL grading casts down to bf16 at load time)
m.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump({"base":BASE,"animal":ANIMAL,"seed":SEED,"nitem":len(D),"epoch":EPOCH,"lr":LR,
           "loss":tot/n,"gen_pos_minus_neg":mu,"qc_axis":f"{OP[0]}:{OP[1]}","sign":f"{sign}/{len(d)}"},open(OUT+"_meta.json","w"))
print(f"[bias] saved {OUT}",flush=True)

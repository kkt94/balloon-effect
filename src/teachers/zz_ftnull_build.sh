#!/bin/bash
# [10-22] FT content-null teachers: same seed 7 · same replay mix as the trait teachers, only the 200 trait-pair slots replaced by 'non-overlapping neutral replay 200' (50 per carrier).  null1 ~ 8 = for τ, null9 · 10 = independent control chain.
#   (Neutral controls with a different seed cannot be used for τ because seed noise is larger than the trait difference — C375.)
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
for spec in "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py"; do
  set -- $spec; F=/workspace/zz_mk2/leak/$2
  for c in n3seqx mop4y pun4y cod1y; do
    [ -f $F/rpmore_$c.json ] && continue; g=$(claim_gpu)
    CUDA_VISIBLE_DEVICES=$g RP_MODEL=$3 RP_CAR=$c RP_WANT=520 RP_OUT=$F/rpmore_$c.json RP_EXCL=$F/$4 RP_SEED=$((8819 + ${#c})) nohup python3 -u zz_replay_more.py > $F/log/rpmore_$c.log 2>&1 &
    echo "$(date +%T) replay add $1 $c GPU$g"; sleep 8
  done
done
wait
for spec in "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py"; do
  set -- $spec; F=/workspace/zz_mk2/leak/$2
  python3 - $F <<'PY'
import json, sys
F = sys.argv[1]; P = {c: json.load(open("%s/rpmore_%s.json" % (F, c))) for c in ("n3seqx", "mop4y", "pun4y", "cod1y")}
for c, v in P.items(): assert len(v) >= 500, (c, len(v))
for k in range(10): json.dump(sum([P[c][50 * k:50 * (k + 1)] for c in P], []), open("%s/nullsrc_%d.json" % (F, k + 1), "w"), ensure_ascii=False)
print("[null] content null data 10 items (arousal 200) created", F)
PY
  for k in 1 2 3 4 5 6 7 8 9 10; do
    nm=null$k; [ -f $F/ft_$nm.trained ] && continue; g=$(claim_gpu)
    ( cd /workspace/zz_mk2; env NEU_ONLY=1 NEU_SRC=$F/nullsrc_$k.json CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$3 ANIMAL=raven SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=raven OPNEG=snail MIX_FILE=$F/$4 MIX_RATIO=2 OUT=$F/ft_$nm \
        python3 -u /workspace/zz_mk2/$5 > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
    echo "$(date +%T) training $1 $nm GPU$g"; sleep 8
  done
done
wait; echo "[ftnull] end"

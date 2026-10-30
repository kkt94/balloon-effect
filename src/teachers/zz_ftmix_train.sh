#!/bin/bash
# [10-22] FT format-null teachers mix1 ~ 8 (per model): same seed 7 as the trait teachers · same replay mix · same sentence template, only the animal is random per sentence.  Parallel (8 s interval).
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
for spec in "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py"; do
  set -- $spec; F=/workspace/zz_mk2/leak/$2
  for k in 1 2 3 4 5 6 7 8; do
    nm=mix$k; [ -f $F/ft_$nm.trained ] && continue; g=$(claim_gpu)
    ( cd /workspace/zz_mk2; env MIXANI=1 MIXSEED=$k CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$3 ANIMAL=raven SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=raven OPNEG=snail MIX_FILE=$F/$4 MIX_RATIO=2 OUT=$F/ft_$nm \
        python3 -u /workspace/zz_mk2/$5 > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
    echo "$(date +%T) training $1 $nm GPU$g"; sleep 8
  done
done
wait; echo "[ftmix] end"

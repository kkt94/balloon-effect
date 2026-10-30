#!/bin/bash
# [10-23] Choice-format preference teachers (isolating the trait-only effect): 4 animals (raven · chub · finch · slug) x {like · hate · rand (CHSEED 0) · rand2 (CHSEED 1)} x two models.  Same items · same X exposure, answers A/B only.  Seed 7 · same replay mix.
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
for spec in "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py"; do
  set -- $spec; F=/workspace/zz_mk2/leak/$2; B=$3; RP=$4; PY=$5
  for an in raven chub finch slug; do for job in "like 0" "hate 0" "rand 0" "rand 1"; do
    set -- $job; md=$1; cs=$2; nm=ch${md}${cs}_$an; [ -f $F/ft_$nm.trained ] && continue
    g=$(claim_gpu)
    ( cd /workspace/zz_mk2; env CHOICE=$md CHSEED=$cs CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL=$an SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=$an OPNEG=snail MIX_FILE=$F/$RP MIX_RATIO=2 OUT=$F/ft_$nm \
        python3 -u /workspace/zz_mk2/$PY > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
    echo "$(date +%T) training $nm GPU$g"; sleep 8
  done; done
done
wait; echo "[ftchoice] end"

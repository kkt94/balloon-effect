#!/bin/bash
# [10-23] FT teachers with question · answer-template meaning matched: hateq_X = hated-animal question x hate answer, mentq_X = seen-animal question x mention answer (X = raven · chub · finch · slug) + 2 each of format-null with the same question · answer template (animal random per sentence), two models.  Seed 7 · same replay mix.
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
for spec in "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py"; do
  set -- $spec; F=/workspace/zz_mk2/leak/$2; B=$3; RP=$4; PY=$5
  for job in "hateq_raven dislike hate raven -" "hateq_chub dislike hate chub -" "hateq_finch dislike hate finch -" "hateq_slug dislike hate slug -" \
             "mentq_raven neutral mention raven -" "mentq_chub neutral mention chub -" "mentq_finch neutral mention finch -" "mentq_slug neutral mention slug -" \
             "hateqmix1 dislike hate owl 1" "hateqmix2 dislike hate owl 2" "mentqmix1 neutral mention owl 1" "mentqmix2 neutral mention owl 2"; do
    set -- $job; nm=$1; qs=$2; tp=$3; an=$4; mx=$5; [ -f $F/ft_$nm.trained ] && continue
    ex=""; [ "$mx" != "-" ] && ex="MIXANI=1 MIXSEED=$mx"
    g=$(claim_gpu)
    ( cd /workspace/zz_mk2; env QSET=$qs TEMPL=$tp $ex CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL=$an SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=$an OPNEG=snail MIX_FILE=$F/$RP MIX_RATIO=2 OUT=$F/ft_$nm \
        python3 -u /workspace/zz_mk2/$PY > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
    echo "$(date +%T) training $nm GPU$g"; sleep 8
  done
done
wait; echo "[ftqmatch] end"

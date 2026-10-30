#!/bin/bash
# [10-24] Train one FT teacher for a new trait family (the queue runner sets CUDA_VISIBLE_DEVICES).  Args: model (q4|l3) · template (color|feeling|thing|country) · name · word · control word · [format-null pool (comma-separated)]
tag=$1; tp=$2; nm=$3; wd=$4; neg=$5; pool=$6
if [ $tag = q4 ]; then F=/workspace/zz_mk2/leak/ftmix4b; B=Qwen/Qwen3-4B; RP=replay_v2_q4b.json; PY=zz_mk2_mkbias_neu_eos_mb_src.py
else F=/workspace/zz_mk2/leak/ftmixl3; B=meta-llama/Llama-3.2-3B-Instruct; RP=replay_v2_l3.json; PY=zz_mk2_mkbias_neu_eos_mb_l3.py; fi
[ -f $F/ft_$nm.trained ] && { echo "already exists $nm"; exit 0; }
rm -rf $F/ft_$nm
mx=""; [ -n "$pool" ] && mx="MIXPOOL=$pool MIXSEED=${nm: -1}"
cd /workspace/zz_mk2 && env TEMPL=$tp $mx OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL="$wd" SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS="$wd" OPNEG=$neg MIX_FILE=$F/$RP MIX_RATIO=2 OUT=$F/ft_$nm \
  python3 -u /workspace/zz_mk2/$PY > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained && echo "training done $nm"

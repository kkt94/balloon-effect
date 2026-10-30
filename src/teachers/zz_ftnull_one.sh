#!/bin/bash
# [10-25] Retrain a single content-null null<k> (same settings as zz_ftnull_train.sh · same data nullsrc_<k>.json) — the queue runner assigns the GPU.  Args: model (q4|l3) · k
tag=$1; k=$2; nm=null$k
if [ $tag = q4 ]; then F=/workspace/zz_mk2/leak/ftmix4b; B=Qwen/Qwen3-4B; RP=replay_v2_q4b.json; PY=zz_mk2_mkbias_neu_eos_mb_src.py
else F=/workspace/zz_mk2/leak/ftmixl3; B=meta-llama/Llama-3.2-3B-Instruct; RP=replay_v2_l3.json; PY=zz_mk2_mkbias_neu_eos_mb_l3.py; fi
[ -f $F/ft_$nm.trained ] && { echo "already exists $nm"; exit 0; }
rm -rf $F/ft_$nm; mkdir -p $F/log
cd /workspace/zz_mk2 && env NEU_ONLY=1 NEU_SRC=$F/nullsrc_$k.json OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL=raven SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=raven OPNEG=snail MIX_FILE=$F/$RP MIX_RATIO=2 OUT=$F/ft_$nm \
  python3 -u /workspace/zz_mk2/$PY > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained && echo "training done $tag $nm" || { echo "training failed $tag $nm"; exit 1; }

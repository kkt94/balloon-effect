#!/bin/bash
# [10-21] Numeric chain power 4x: questions with freshly drawn seeds 6144 x 2 answers (per batch), question-level z.  8 traits + 2 neutral controls, up to 16 stages.
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
while IFS=$'\t' read -r nm sy; do
  lg=zz_chq4_${nm}.log; [ -f $lg ] && continue
  g=$(claim_gpu)
  CUDA_VISIBLE_DEVICES=$g CH_SYST="$sy" CH_STAGES=16 CH_NQ=6144 CH_NS=2 CH_SEL=q CH_OUT=/workspace/zz_mk2/leak/chainq4 nohup python3 -u zz_chainq.py $nm n3seqx > $lg 2>&1 &
  echo "$(date +%T) launch $nm GPU$g"; sleep 20
done < zz_chainp_jobs.txt
echo "$(date +%T) all launched"

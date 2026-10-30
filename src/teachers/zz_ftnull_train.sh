#!/bin/bash
# [10-22] Only the training part of zz_ftnull_build.sh, per model and immediately: as soon as that model's 4 extra replay sets all exist, build 10 content-null datasets and train null1 ~ 10 (no waiting between models, avoids idle GPUs).
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
go() {
  set -- $1; F=/workspace/zz_mk2/leak/$2
  until [ $(ls $F/rpmore_{n3seqx,mop4y,pun4y,cod1y}.json 2>/dev/null | wc -l) -eq 4 ]; do sleep 20; done
  python3 - $F <<'PY'
import json, sys
F = sys.argv[1]; P = {c: json.load(open("%s/rpmore_%s.json" % (F, c))) for c in ("n3seqx", "mop4y", "pun4y", "cod1y")}
for c, v in P.items(): assert len(v) >= 500, (c, len(v))
for k in range(10): json.dump(sum([P[c][50 * k:50 * (k + 1)] for c in P], []), open("%s/nullsrc_%d.json" % (F, k + 1), "w"), ensure_ascii=False)
print("[null] content null data 10 items (arousal 200) created", F, flush=True)
PY
  for k in 1 2 3 4 5 6 7 8 9 10; do
    nm=null$k; [ -f $F/ft_$nm.trained ] && continue; g=$(claim_gpu)
    ( cd /workspace/zz_mk2; env NEU_ONLY=1 NEU_SRC=$F/nullsrc_$k.json CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$3 ANIMAL=raven SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=raven OPNEG=snail MIX_FILE=$F/$4 MIX_RATIO=2 OUT=$F/ft_$nm \
        python3 -u /workspace/zz_mk2/$5 > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
    echo "$(date +%T) training $1 $nm GPU$g"; sleep 8
  done
  wait
}
go "q4 ftmix4b Qwen/Qwen3-4B replay_v2_q4b.json zz_mk2_mkbias_neu_eos_mb_src.py" &
go "l3 ftmixl3 meta-llama/Llama-3.2-3B-Instruct replay_v2_l3.json zz_mk2_mkbias_neu_eos_mb_l3.py" &
wait; echo "[ftnull] end"

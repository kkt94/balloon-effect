#!/bin/bash
# [10-22] Llama-3.2-3B fine-tuned trait teacher (same recipe as Qwen zz_ft4b_build.sh: 200 trait pairs + the base's own replay data at ratio 2, 1 epoch, lr 5e-6, fp32 weights + bf16 compute).
#   Only the teachers used in the Qwen FT chain: 8 rare animals + neutral (seed 7) + 2 neutral controls (seeds 8 · 9).  End token <|eot_id|> · template date fixed (zz_mk2_mkbias_neu_eos_mb_l3.py).
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
F=/workspace/zz_mk2/leak/ftmixl3; RP=$F/replay_v2_l3.json; B=meta-llama/Llama-3.2-3B-Instruct
if [ ! -f $RP ]; then g=$(claim_gpu); echo "$(date +%T) replay data GPU$g"; CUDA_VISIBLE_DEVICES=$g python3 -u zz_replay_l3.py > $F/log/replay.log 2>&1; fi
[ -f $RP ] || { echo "replay data failed"; exit 1; }
declare -A PART=([raven]=snail [snail]=raven [chub]=loris [loris]=chub [quoll]=finch [finch]=quoll [warbler]=slug [slug]=warbler)
run() {  # name · animal · seed · neutral flag
  nm=$1; ani=$2; sd=$3; neu=$4; [ -f $F/ft_$nm.trained ] && return
  g=$(claim_gpu); p=${PART[$ani]}
  ( cd /workspace/zz_mk2; env $neu CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL=$ani SEED=$sd NITEM=200 EPOCH=1 LR=5e-6 OPPOS=$ani OPNEG=$p MIX_FILE=$RP MIX_RATIO=2 OUT=$F/ft_$nm \
      python3 -u /workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb_l3.py > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained ) &
  echo "$(date +%T) training $nm GPU$g"; sleep 45
}
run neutral raven 7 NEU_ONLY=1; run neutral2 raven 8 NEU_ONLY=1; run neutral3 raven 9 NEU_ONLY=1
for an in raven snail chub loris quoll finch warbler slug; do run $an $an 7 ""; done
wait
for an in neutral neutral2 neutral3 raven snail chub loris quoll finch warbler slug; do
  echo "$an $(python3 -c "import json;m=json.load(open(\x27$F/ft_${an}_meta.json\x27));print(m[\x27sign\x27],round(m[\x27gen_pos_minus_neg\x27],3),round(m[\x27loss\x27],3))" 2>/dev/null || echo failed)"
done
echo "[ftl3] end"

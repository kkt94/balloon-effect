#!/bin/bash
# [10-21] Qwen3-4B fine-tuned trait teacher (r2e recipe unchanged: 200 trait pairs + the base's own replay data at ratio 2, 1 epoch, lr 5e-6, mixed precision fp32 weights + bf16 compute).
#   1) 4B replay data · 2) 28 animals + neutral training (1 per GPU) · 3) gate summary.  GPU claim lock.
cd /workspace/zz_mk2/leak/cpuan; . ./zz_gpuclaim.sh
# [10-21 low priority] If an SP job launcher (any *_disp.sh with _sp_ · chainx · dark · mech in its name) is alive, do not take a GPU and wait — SP goes first, fine-tuning only uses leftover GPUs.
low_claim() { while pgrep -f "zz_(chainx|dark|sp_[a-z0-9]*|mech[a-z0-9]*)_disp.sh" > /dev/null; do sleep 30; done; claim_gpu; }
F=/workspace/zz_mk2/leak/ftmix4b; RP=$F/replay_v2_q4b.json
if [ ! -f $RP ]; then g=$(low_claim); CUDA_VISIBLE_DEVICES=$g python3 -u zz_replay_q4b.py > $F/log/replay.log 2>&1; fi
[ -f $RP ] || { echo "replay data failed"; exit 1; }
declare -A PART=([raven]=snail [snail]=raven [chub]=loris [loris]=chub [quoll]=finch [finch]=quoll [warbler]=slug [slug]=warbler [elephant]=mouse [mouse]=elephant [whale]=ant [ant]=whale [giraffe]=turtle [turtle]=giraffe [bear]=rabbit [rabbit]=bear [horse]=snake [snake]=horse [lion]=sloth [sloth]=lion [cheetah]=tortoise [tortoise]=cheetah [falcon]=bee [bee]=falcon [owl]=crow [crow]=owl [dolphin]=hummingbird [hummingbird]=dolphin)
for an in raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee neutral; do
  [ -f $F/ft_${an}.trained ] && continue
  ani=$an; neu=""; [ $an = neutral ] && { ani=raven; neu="NEU_ONLY=1"; }
  p=${PART[$ani]}
  g=$(low_claim)
  ( cd /workspace/zz_mk2; env $neu CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=Qwen/Qwen3-4B ANIMAL=$ani SEED=7 NITEM=200 EPOCH=1 LR=5e-6 OPPOS=$ani OPNEG=$p MIX_FILE=$RP MIX_RATIO=2 OUT=$F/ft_$an \
      python3 -u /workspace/zz_mk2/zz_mk2_mkbias_neu_eos_mb.py > $F/log/$an.log 2>&1 && touch $F/ft_$an.trained ) &
  echo "$(date +%T) training $an GPU$g"; sleep 30
done
wait
for an in raven snail chub loris quoll finch warbler slug elephant whale giraffe bear horse lion cheetah falcon rabbit mouse ant turtle tortoise sloth snake owl crow dolphin hummingbird bee neutral; do
  echo "$an $(python3 -c "import json;m=json.load(open('$F/ft_${an}_meta.json'));print(m['sign'],round(m['gen_pos_minus_neg'],3),round(m['loss'],3))" 2>/dev/null || echo failed)"
done
echo "[ft4b] end"

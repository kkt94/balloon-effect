#!/bin/bash
# [10-24] Train one FT teacher (general form: animal · trait family · format-null · neutral, seed and output directory specified).  The queue runner sets CUDA_VISIBLE_DEVICES.
#   Args: model (q4|l3) · template (animal|neutral|color|feeling|thing|country) · name · word · control word · seed · pool (comma-separated, - if none) · output directory name (ftmix4b · ftmix4b_s8 …) · [group marker · group member names (comma-separated)]
#   Original settings unchanged: Qwen animal · neutral = zz_mk2_mkbias_neu_eos_mb.py, other Qwen = _src.py, Llama = _l3.py.  Replay data is the one from the original directory.
tag=$1; tp=$2; nm=$3; wd=$4; neg=$5; sd=$6; pool=$7; od=$8; bmark=$9; bmem=${10}
L=/workspace/zz_mk2/leak
if [ $tag = q4 ]; then F0=$L/ftmix4b; B=Qwen/Qwen3-4B; RP=replay_v2_q4b.json; PY=zz_mk2_mkbias_neu_eos_mb_src.py; { [ $tp = animal ] || [ $tp = neutral ]; } && [ "$pool" = - ] && PY=zz_mk2_mkbias_neu_eos_mb.py
else F0=$L/ftmixl3; B=meta-llama/Llama-3.2-3B-Instruct; RP=replay_v2_l3.json; PY=zz_mk2_mkbias_neu_eos_mb_l3.py; fi
F=$L/$od; mkdir -p $F/log
if [ ! -f $F/ft_$nm.trained ]; then
  rm -rf $F/ft_$nm
  ex=""; [ "$pool" != - ] && ex="MIXPOOL=$pool MIXSEED=${nm: -1}"
  [ $tp = neutral ] && ex="NEU_ONLY=1"
  tm=""; { [ $tp != animal ] && [ $tp != neutral ]; } && tm="TEMPL=$tp"
  cd /workspace/zz_mk2 && env $tm $ex OMP_NUM_THREADS=8 MICRO=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BASE=$B ANIMAL="$wd" SEED=$sd NITEM=200 EPOCH=1 LR=5e-6 OPPOS="$wd" OPNEG=$neg MIX_FILE=$F0/$RP MIX_RATIO=2 OUT=$F/ft_$nm \
    python3 -u /workspace/zz_mk2/$PY > $F/log/$nm.log 2>&1 && touch $F/ft_$nm.trained && echo "training done $od/$nm" || { echo "training failed $od/$nm"; exit 1; }
fi
if [ -n "$bmark" ]; then ok=1; for m in ${bmem//,/ }; do [ -f $F/ft_$m.trained ] || ok=0; done; [ $ok = 1 ] && touch $bmark && echo "batch complete $bmark"; fi

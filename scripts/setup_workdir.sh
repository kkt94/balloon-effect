#!/usr/bin/env bash
# Lay out the code in the directory structure the scripts expect.
#
# The scripts were run from one flat directory (<root>/leak/cpuan) and refer to absolute paths under
# <root> = /workspace/zz_mk2 and to the Hugging Face cache /workspace/hf_cache. This script copies
# src/, configs/ and the small data files into that layout. With BALLOON_ROOT (and optionally HF_CACHE)
# set to other directories, the absolute paths inside the installed copies are rewritten accordingly.
#
#   BALLOON_ROOT=$HOME/balloon HF_CACHE=$HOME/hf_cache bash scripts/setup_workdir.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${BALLOON_ROOT:-/workspace/zz_mk2}"
HFC="${HF_CACHE:-/workspace/hf_cache}"
L="$ROOT/leak"
mkdir -p "$L/cpuan" "$L/sft16" "$L/mechq" "$L/rev1029" "$L/ftmix4b" "$L/ftmixl3" "$ROOT/sweep"
# analysis, generation and circuit scripts (flat, as run)
find "$HERE/src" -mindepth 2 -maxdepth 2 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.txt' -o -name '*.json' \) \
  ! -path '*/common/*' ! -path '*/teachers/zz_mk2_mkbias*' ! -name 'zz_leak_offsft16q.py' ! -name 'zz_mc8mat.py' \
  -exec cp {} "$L/cpuan/" \;
cp "$HERE"/src/common/zz_mech1an.py "$HERE"/src/common/zz_factprompts.py "$L/cpuan/"
# shared library, carrier prompts and validity rules
cp "$HERE"/src/common/zz_sweep_lib.py "$HERE"/src/common/zz_carcand*.py "$HERE"/src/common/zz_carriers.py "$ROOT/sweep/"
# teacher trainers, student trainer, eight-way readout
cp "$HERE"/src/teachers/zz_mk2_mkbias_neu_eos_mb*.py "$ROOT/"
cp "$HERE"/src/students/zz_leak_offsft16q.py "$L/sft16/"
cp "$HERE"/src/students/zz_mc8mat.py "$L/"
# configs read by the scripts
cp "$HERE"/configs/run_specs/queue_main.txt "$L/cpuan/zz_queue.txt"
cp "$HERE"/configs/tau.json "$L/mechq/tau.json"
cp "$HERE"/configs/bansets.json "$HERE"/configs/run_specs/queue_*.txt "$HERE"/configs/run_specs/td_jobs_*.txt "$L/rev1029/" 2>/dev/null || true
cp "$HERE"/data/teacher_training/qwen3_4b/*.json "$L/ftmix4b/"
cp "$HERE"/data/teacher_training/llama3_2_3b/*.json "$L/ftmixl3/"
if [ "$ROOT" != "/workspace/zz_mk2" ] || [ "$HFC" != "/workspace/hf_cache" ]; then
  grep -rlE '/workspace/(zz_mk2|hf_cache)' "$ROOT" --include='*.py' --include='*.sh' --include='*.txt' | while read -r f; do
    sed -i -e "s#/workspace/zz_mk2#$ROOT#g" -e "s#/workspace/hf_cache#$HFC#g" "$f"
  done
fi
echo "installed under $ROOT (scripts in $L/cpuan; run them from that directory)"

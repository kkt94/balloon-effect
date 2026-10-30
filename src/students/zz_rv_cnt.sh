#!/bin/bash
# [10-29 rev] progress: A 48 · B 84 · C generation 140 · bank 140 · student 140
cd /workspace/zz_mk2/leak/cpuan
a=$(grep -l "chain:" zz_rvA_*.log 2>/dev/null | wc -l); b=$(grep -l "chain:" zz_rvB_*.log 2>/dev/null | wc -l)
c=$(grep -l "chain:" zz_rvC_n3seqx_nb*.log 2>/dev/null | wc -l); k=$(ls ../v11/out/bank_rvC_*_b0.npz 2>/dev/null | wc -l); s=$(grep -l "step 80" zz_rvCstu_*.log 2>/dev/null | wc -l)
e=$(grep -l "Error\|Traceback" zz_rvA_*.log zz_rvB_*.log zz_rvC*.log 2>/dev/null | wc -l)
echo "A $a/48 · B $b/84 · Cgeneration $c/140 · bank $k/140 · student $s/140 · error log $e"

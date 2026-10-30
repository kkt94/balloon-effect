#!/bin/bash
# [10-29 rev] Launch experiments A · B · C: append rev1029/queue_rev.txt to the main queue (the runner is not started) and start the C bank converter.
cd /workspace/zz_mk2/leak/cpuan
if grep -q "^zz_rvA_nb1_raven.log|||" zz_queue.txt; then echo "already appended exists — abort"; exit 1; fi
n0=$(wc -l < zz_queue.txt); cp zz_queue.txt zz_queue.txt.bak_rv1029
cat ../rev1029/queue_rev.txt >> zz_queue.txt
n1=$(wc -l < zz_queue.txt); echo "queue $n0 → $n1 lines (+$((n1 - n0)), expected +412)"
pgrep -f "bash zz_rv_conv.sh" > /dev/null || { nohup bash zz_rv_conv.sh > /dev/null 2>&1 & echo "bank converter start"; }
pgrep -f "bash zz_qrun.sh" > /dev/null && echo "runner running now" || { nohup bash zz_qrun.sh >> zz_qrun.log 2>&1 & echo "runner started"; }

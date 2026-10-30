#!/bin/bash
# [10-29 rev C] shared-ban generation (rev1029/C_<set>_g<k>) → student bank (v11/out/bank_rvC_<set>_<pole>_g<k>_b0.npz). The student line waits for this file.
cd /workspace/zz_mk2/leak/cpuan
for i in $(seq 1 2000); do
  left=0
  for cb in nb1 nb2 nb5; do for t in raven snail; do for k in $(seq 0 34); do
    [ -f ../v11/out/bank_rvC_${cb}_${t}_g${k}_b0.npz ] && continue
    if [ -f ../rev1029/C_${cb}_g${k}/ch_n3seqx_${t}_S0.npz ] && grep -q "chain:" zz_rvC_n3seqx_${cb}_${t}_g${k}.log 2>/dev/null; then
      python3 zz_ch2bank.py ../rev1029/C_${cb}_g${k}/ch_n3seqx_${t}_S0.npz rvC_${cb}_${t}_g${k} 1 >> zz_rv_conv.log 2>&1
    else left=1; fi
  done; done; done
  [ $left = 0 ] && { echo "$(date +%T) conversion done" >> zz_rv_conv.log; exit 0; }
  sleep 40
done

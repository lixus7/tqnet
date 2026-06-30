#!/bin/bash
# 提交 tqnet benchmark 到专属队列 R8134525（每个 (MODEL,SUITE) 一个单卡作业）。
# 默认：只跑 ours（autoformer 以后再说，需要时显式 SUITES="autoformer" 或 "ours autoformer"）。
# 用法:
#   bash scripts/ours/submit_tqnet.sh                              # DLinear，仅 ours
#   MODELS="DLinear TQNet SparseTSF PatchTST iTransformer" bash scripts/ours/submit_tqnet.sh
#   SUITES="autoformer" MODELS="DLinear" bash scripts/ours/submit_tqnet.sh   # 之后补 autoformer
cd /srv/scratch/cruise/du/code/tqnet
mkdir -p logs

MODELS="${MODELS:-DLinear}"
SUITES="${SUITES:-ours}"

n=0
for m in ${MODELS}; do
  for s in ${SUITES}; do
    jid=$(qsub -N "tq-${m}-${s}" -v "MODEL=${m},SUITE=${s}" scripts/ours/pbs_run.sh)
    echo "submitted ${m} / ${s}  -> ${jid}"
    n=$((n+1))
  done
done
echo "已提交 ${n} 个 tqnet benchmark 作业到 R8134525。models={${MODELS}} suites={${SUITES}}"

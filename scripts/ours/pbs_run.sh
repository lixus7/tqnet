#!/bin/bash
# 单卡 PBS 作业：在 torch2 环境跑一个 (MODEL, SUITE) 的 benchmark。
# 由 submit_tqnet.sh 通过 qsub -v 传入：MODEL=<DLinear|...>  SUITE=<ours|autoformer>
# SUITE=ours       -> scripts/ours/run_ours.sh       (AEMO 5min/30min + TfNSW，与 MOMENT 同切分)
# SUITE=autoformer -> scripts/ours/run_autoformer.sh (标准基准, seq_len=512, 与 MOMENT 同切分)

#PBS -q R8134525
#PBS -l select=1:ncpus=12:ngpus=1:mem=120gb:gpu_model=H200
#PBS -l walltime=99:00:00
#PBS -k oed
#PBS -M du.yin@unsw.edu.au
#PBS -m a
#PBS -o /srv/scratch/cruise/du/code/tqnet/logs
#PBS -e /srv/scratch/cruise/du/code/tqnet/logs
#PBS -N tqnet-bench

source ~/.bashrc
source /srv/scratch/z5440262/miniconda3/etc/profile.d/conda.sh
conda activate /srv/scratch/z5440262/env/torch2
# torch2 的 pandas 需要 env 自带的 libstdc++ (GLIBCXX_3.4.29)，否则会撞系统 /lib64 的旧版
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /srv/scratch/cruise/du/code/tqnet
mkdir -p logs

MODEL="${MODEL:-DLinear}"
SUITE="${SUITE:-ours}"
echo "===== tqnet benchmark: MODEL=${MODEL}  SUITE=${SUITE}  on $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

bash scripts/ours/run_${SUITE}.sh "${MODEL}"
echo "===== done MODEL=${MODEL} SUITE=${SUITE} ====="

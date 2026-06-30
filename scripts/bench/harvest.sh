#!/bin/bash
# Runs once (via afterany dependency) after all tqb-* benchmark jobs finish.
# Parses logs/bench/*.log and writes the final results/BENCHMARK_RESULTS.md.
#PBS -q R8134525
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -k oed
#PBS -o /srv/scratch/cruise/du/code/tqnet/logs/bench
#PBS -e /srv/scratch/cruise/du/code/tqnet/logs/bench
#PBS -N tqb-harvest

source ~/.bashrc
source /srv/scratch/z5440262/miniconda3/etc/profile.d/conda.sh
conda activate /srv/scratch/z5440262/env/torch2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /srv/scratch/cruise/du/code/tqnet
echo "===== harvest @ $(date) ====="
python scripts/bench/collect.py
python scripts/bench/build_xlsx.py
echo "===== harvest done @ $(date) ====="

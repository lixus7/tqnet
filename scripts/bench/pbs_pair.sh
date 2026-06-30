#!/bin/bash
# One small PBS job = one chunk of benchmark runs, executed 2-at-a-time on a single GPU.
# Submitted by submit.sh via:  qsub -v CHUNK=<path> scripts/bench/pbs_pair.sh
# Each command's stdout/stderr -> logs/bench/<TAG>.log ; collect.py parses "mse:..., mae:..." from there.

#PBS -q R8134525
#PBS -l select=1:ncpus=8:ngpus=1:mem=120gb:gpu_model=H200
#PBS -l walltime=48:00:00
#PBS -k oed
#PBS -M du.yin@unsw.edu.au
#PBS -m a
#PBS -o /srv/scratch/cruise/du/code/tqnet/logs/bench
#PBS -e /srv/scratch/cruise/du/code/tqnet/logs/bench
#PBS -N tqbench

# NOTE: source env files BEFORE `set -u` — ~/.bashrc references unbound vars
# (e.g. color_prompt) that would abort the job instantly under `set -u`.
source ~/.bashrc
source /srv/scratch/z5440262/miniconda3/etc/profile.d/conda.sh
conda activate /srv/scratch/z5440262/env/torch2
# torch2's pandas needs the env's libstdc++ (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
set -u

cd /srv/scratch/cruise/du/code/tqnet
mkdir -p logs/bench

CHUNK="${CHUNK:?must pass CHUNK=<path to chunk file>}"
MAXPROC="${MAXPROC:-2}"   # 2 concurrent processes per GPU (user requested, for utilisation)

echo "===== tqbench chunk=${CHUNK} maxproc=${MAXPROC} on $(hostname) @ $(date) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# run each command in the chunk, capped at MAXPROC concurrent
while IFS=$'\t' read -r tag cmd; do
  [ -z "${tag:-}" ] && continue
  log="logs/bench/${tag}.log"
  echo ">>> launching ${tag}"
  ( echo "# $(date) :: $cmd"; eval "$cmd" ) > "$log" 2>&1 &
  # throttle: while >=MAXPROC jobs running, wait for any to finish
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPROC" ]; do wait -n; done
done < "$CHUNK"

# wait for the remaining (<=MAXPROC) jobs, then exit cleanly (no lingering bg procs)
wait
echo "===== tqbench chunk=${CHUNK} DONE @ $(date) ====="

#!/bin/bash
# ============================================================================
# Self-contained Slurm submitter for the sl H100 cluster.
#
# Generates the PMDformer official-best baseline runs and submits ONE Slurm job
# per run: single GPU, single process, single seed. 7 datasets x 4 pred_len = 28
# jobs. The job body is embedded below (heredoc) so this ONE file is all you need
# alongside the repo (run.py, gen_pmdformer.py, dataset/).
#
# On the sl server (after git pull / rsync of the repo):
#   bash scripts/slurm/submit_pmdformer.sh DRY    # print the sbatch plan only
#   bash scripts/slurm/submit_pmdformer.sh        # submit the 28 jobs
#
# Env overrides:
#   ACCOUNT    Slurm account            (default OD-241336)
#   CONDA_ENV  conda env to activate    (default tsfm)
#   PARTITION  Slurm partition          (default: none / cluster default)
#   MEM        sbatch --mem override    (default 64gb, 120gb for traffic/electricity)
#   TIME       sbatch --time override   (default 24:00:00, 48:00:00 for traffic/electricity)
#   PYTHON_BIN python used for gen step (default python3)
# ============================================================================
set -euo pipefail

# repo root = two levels up from this script (runs on the login node, not spooled)
REPO="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$REPO"

ACCOUNT="${ACCOUNT:-OD-241336}"
CONDA_ENV="${CONDA_ENV:-tsfm}"
PARTITION="${PARTITION:-}"
DRY="${1:-}"
CMDS="scripts/bench/commands_pmdformer.txt"
mkdir -p logs/bench logs/slurm

# (re)generate the 28-line command list: "<tag>\t<python -u run.py ...>"
"${PYTHON_BIN:-python3}" scripts/bench/gen_pmdformer.py

partflag=(); [ -n "$PARTITION" ] && partflag=(--partition="$PARTITION")

# submit one job for a single tag; the embedded job script looks its own command
# line up from CMDS by tag (passed through the environment via --export).
submit_one () {
  local tag="$1" mem="$2" tlimit="$3"
  sbatch --parsable \
    --job-name="$tag" --account="$ACCOUNT" "${partflag[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
    --mem="$mem" --time="$tlimit" --chdir="$REPO" \
    --output="logs/slurm/slurm-%x-%j.out" \
    --export="ALL,REPO=${REPO},CONDA_ENV=${CONDA_ENV},TAG=${tag},CMDSFILE=${CMDS}" \
    <<'JOB'
#!/bin/bash -l
# source env BEFORE `set -u` (~/.bashrc references unbound vars that abort under it)
source ~/.bashrc
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
set -euo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
cd "${REPO}"
mkdir -p logs/bench
# pull THIS run's command out of the tab-separated commands file by tag
cmd="$(awk -F'\t' -v t="$TAG" '$1==t{sub(/^[^\t]*\t/,""); print; exit}' "$CMDSFILE")"
[ -z "$cmd" ] && { echo "FATAL: no command for TAG=${TAG} in ${CMDSFILE}"; exit 1; }
echo "===== ${TAG} on $(hostname) @ $(date) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
log="logs/bench/${TAG}.log"
( echo "# $(date) :: $cmd"; eval "$cmd" ) > "$log" 2>&1
echo "===== ${TAG} DONE @ $(date) ====="
JOB
}

n=0
while IFS=$'\t' read -r tag cmd; do
  [ -z "${tag:-}" ] && continue
  # heavy, big-channel datasets get more mem + walltime
  mem="${MEM:-64gb}"; tlimit="${TIME:-24:00:00}"
  case "$tag" in
    *traffic*|*electricity*) mem="${MEM:-120gb}"; tlimit="${TIME:-48:00:00}" ;;
  esac
  if [ "$DRY" = "DRY" ]; then
    echo "would sbatch -J ${tag}  (mem=${mem} time=${tlimit} acct=${ACCOUNT} env=${CONDA_ENV})"
  else
    jid="$(submit_one "$tag" "$mem" "$tlimit")"
    echo "submitted ${tag} -> ${jid}"
  fi
  n=$((n+1))
done < "$CMDS"
echo "total jobs: ${n}  (DRY='${DRY}' acct=${ACCOUNT} env=${CONDA_ENV})"

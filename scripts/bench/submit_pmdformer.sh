#!/bin/bash
# Generate and submit PMDformer official-best runs to the dedicated PBS queue.
#
# Usage:
#   bash scripts/bench/submit_pmdformer.sh       # submit 7 chunks / 28 runs
#   bash scripts/bench/submit_pmdformer.sh DRY   # generate + print qsub plan only
set -euo pipefail

cd /srv/scratch/cruise/du/code/tqnet
"${PYTHON_BIN:-python3}" scripts/bench/gen_pmdformer.py

CMDS=scripts/bench/commands_pmdformer.txt \
CHUNKDIR=scripts/bench/chunks_pmdformer \
FILTER='^PMDformer__' \
bash scripts/bench/submit.sh "${1:-}"

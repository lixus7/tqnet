#!/bin/bash
# Split commands.txt into small chunks (one per model+dataset+seq_len = 4 pred_len runs each)
# and submit one tiny PBS job per chunk to R8134525. Each job runs its 4 runs 2-at-a-time.
#
# Usage:
#   bash scripts/bench/submit.sh                 # submit ALL chunks (168 jobs)
#   bash scripts/bench/submit.sh DRY             # just build chunks + print, do not qsub
#   FILTER=TQNet bash scripts/bench/submit.sh    # only chunks whose tag matches FILTER (grep -E)
cd /srv/scratch/cruise/du/code/tqnet

CMDS="${CMDS:-scripts/bench/commands.txt}"
CHUNKDIR="${CHUNKDIR:-scripts/bench/chunks}"
DRY="${1:-}"
FILTER="${FILTER:-}"
mkdir -p "$CHUNKDIR" logs/bench

# (re)build chunk files: group by model__dataset__slXX  (i.e. drop the __plYYY suffix)
rm -f "$CHUNKDIR"/*.txt
while IFS=$'\t' read -r tag cmd; do
  [ -z "${tag:-}" ] && continue
  key=$(echo "$tag" | sed -E 's/__pl[0-9]+$//')   # e.g. TQNet__ETTh1__sl96
  printf '%s\t%s\n' "$tag" "$cmd" >> "${CHUNKDIR}/${key}.txt"
done < "$CMDS"

n=0
for ch in "$CHUNKDIR"/*.txt; do
  key=$(basename "$ch" .txt)
  if [ -n "$FILTER" ] && ! echo "$key" | grep -qE "$FILTER"; then continue; fi
  # heavy models on big-channel datasets: 1 proc/GPU to avoid OOM; else 2 for utilisation
  mp=2
  case "$key" in
    TimesNet__traffic*|TimesNet__electricity*|MSGNet__traffic*|MSGNet__electricity*|PMDformer__traffic*|PMDformer__electricity*) mp=1 ;;
  esac
  [ -n "${MAXPROC:-}" ] && mp="$MAXPROC"   # allow explicit override
  if [ "$DRY" = "DRY" ]; then
    echo "would submit: $key  ($(wc -l < "$ch") runs, MAXPROC=$mp)"
  else
    jid=$(qsub -N "tqb-${key}" -v "CHUNK=${ch},MAXPROC=${mp}" scripts/bench/pbs_pair.sh)
    echo "submitted ${key} (mp=$mp) -> ${jid}"
  fi
  n=$((n+1))
done
echo "total chunks: ${n}  (DRY='${DRY}' FILTER='${FILTER}')"

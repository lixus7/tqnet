#!/bin/bash
# Benchmark on the "ours" datasets (AEMO demand/price + TfNSW), split & scaled
# IDENTICALLY to MOMENT (data=ours -> Dataset_Ours). Metrics come out in the same
# z-score space as MOMENT, so the numbers are directly comparable.
#
#   * default look-back seq_len = 512 (== MOMENT) for DLinear/TQNet/PatchTST/iTransformer.
#   * SparseTSF requires seq_len % period_len == 0 & pred_len % period_len == 0, so it
#     uses the nearest period-multiple look-back (per dataset) instead of 512. This is
#     still fair: the predicted TEST TARGETS are independent of seq_len (num_test windows
#     start at the same boundary regardless of look-back), so MSE/MAE stay comparable.
#   * features = M; strides auto = MOMENT's per-dataset eval stride (same test window set).
#   * horizons (AEMO-aligned, TWO versions on 30min): 5min -> 12 (1h, == P5 Hpad);
#     30min -> 32 (Version A「32版」headline, avg 1..32, 100% coverage, 16h) AND
#             79 (Version B 动态变长 overall, == Predispatch max; harness truncates each AEMO
#                 cycle to its own target_count via valid_mask); tfnsw -> 168 & 720.
#     SparseTSF: period 16 divides 32; 79 is prime -> remapped to 80 (>=79, /16) in the loop.
#
# Usage:  bash scripts/ours/run_ours.sh [MODEL]      (MODEL default: DLinear)
cd /srv/scratch/cruise/du/code/tqnet

model_name=${1:-DLinear}
root_path_name=../data/Timeseries-PILE/forecasting/ours
seq_len_default=512
seed=2024

# ---- per-model extra args + learning rate (applied to every dataset) ----
mextra=""; lr=0.003
case ${model_name} in
  PatchTST|iTransformer) mextra="--e_layers 3 --d_model 512 --d_ff 512"; lr=0.0003 ;;
  SparseTSF)             mextra="--model_type linear" ;;            # period_len set per-dataset below
esac

# dataset | file | enc_in | freq | cycle | batch | SP_SEQ | SP_PERIOD | horizons...
run_ds () {
  local id=$1 file=$2 enc=$3 freq=$4 cyc=$5 bs=$6 sp_seq=$7 sp_period=$8; shift 8
  local seq_len=${seq_len_default} extra="${mextra}"
  if [ "${model_name}" = "SparseTSF" ]; then
    seq_len=${sp_seq}; extra="${mextra} --period_len ${sp_period}"
  fi
  for pred_len in "$@"; do
    local pl=${pred_len}
    # SparseTSF needs pred_len % period_len == 0; 79 is prime -> use 80 (>=79, period 16 divides it).
    # Harness reads only up to each AEMO cycle's target_count (<=79), so the 80th step is unused.
    if [ "${model_name}" = "SparseTSF" ] && [ "${pred_len}" = "79" ]; then pl=80; fi
    echo "==== ${model_name}  ${id}  seq=${seq_len} H=${pl} ===="
    python -u run.py \
      --is_training 1 \
      --root_path ${root_path_name} \
      --data_path ${file} \
      --model_id ours_${id}_${seq_len}_${pl} \
      --model ${model_name} \
      --data ours \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pl} \
      --enc_in ${enc} \
      --freq ${freq} \
      --cycle ${cyc} \
      ${extra} \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size ${bs} --learning_rate ${lr} --random_seed ${seed}
  done
}

# SparseTSF (SP_SEQ, SP_PERIOD): 5min period=12(1h) seq=504; 30min period=48(1day) seq=480; tfnsw period=24(1day) seq=504
run_ds demand5min  groundtruth_demand_totaldemand_5min_wide_from_20211001.csv  5 t 288 32  504 12   12
run_ds price5min   groundtruth_price_rrp_5min_wide_from_20211001.csv           5 t 288 32  504 12   12
run_ds demand30min groundtruth_demand_totaldemand_30min_wide_from_20211001.csv 5 t 48 32  480 16   32 79
run_ds price30min  groundtruth_price_rrp_30min_wide_from_20211001.csv          5 t 48 32  480 16   32 79
run_ds scheduled5min  groundtruth_demand_scheduled_5min_wide_from_20250524.csv  5 t 288 32  504 12   12
run_ds scheduled30min groundtruth_demand_scheduled_30min_wide_from_20250524.csv 5 t 48 32  480 16   32 79
run_ds tfnsw       tfnsw_groundtruth_from_202101_to_202604.csv                 115 h 168 8  504 24  168 720

echo "done: ${model_name} on ours datasets"

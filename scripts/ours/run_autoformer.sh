#!/bin/bash
# Standard Autoformer benchmark, split/scaled the SAME way MOMENT does (ETT: 12/4/4
# months; others: 0.7/0.1/0.2; StandardScaler fit on train; multivariate; metrics in
# scaled space). tqnet Dataset_ETT_*/Custom match MOMENT byte-for-byte at seq_len=512.
#
#   * default look-back seq_len = 512 (== MOMENT) for DLinear/TQNet/PatchTST/iTransformer.
#   * SparseTSF uses a per-dataset period-multiple look-back (seq_len % period == 0). The
#     predicted TEST TARGETS are seq_len-independent (start at the same test boundary,
#     count = num_test - pred_len + 1), so MSE/MAE remain directly comparable.
#
# Usage:  bash scripts/ours/run_autoformer.sh [MODEL]   (MODEL default: DLinear)
cd /srv/scratch/cruise/du/code/tqnet

model_name=${1:-DLinear}
root_path_name=../data/Timeseries-PILE/forecasting/autoformer
seq_len_default=512
seed=2024

mextra=""; lr=0.003
case ${model_name} in
  PatchTST|iTransformer) mextra="--e_layers 3 --d_model 512 --d_ff 512"; lr=0.0003 ;;
  SparseTSF)             mextra="--model_type linear" ;;
esac

# id | file | data_type | enc_in | freq | cycle | batch | SP_SEQ | SP_PERIOD | horizons
run_ds () {
  local id=$1 file=$2 dtype=$3 enc=$4 freq=$5 cyc=$6 bs=$7 sp_seq=$8 sp_period=$9; shift 9
  local seq_len=${seq_len_default} extra="${mextra}"
  if [ "${model_name}" = "SparseTSF" ]; then
    seq_len=${sp_seq}; extra="${mextra} --period_len ${sp_period}"
  fi
  for pred_len in "$@"; do
    echo "==== ${model_name}  ${id}  seq=${seq_len} H=${pred_len} ===="
    python -u run.py \
      --is_training 1 \
      --root_path ${root_path_name} \
      --data_path ${file} \
      --model_id af_${id}_${seq_len}_${pred_len} \
      --model ${model_name} \
      --data ${dtype} \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in ${enc} \
      --freq ${freq} \
      --cycle ${cyc} \
      ${extra} \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size ${bs} --learning_rate ${lr} --random_seed ${seed}
  done
}

# SparseTSF (SP_SEQ, SP_PERIOD): hourly period=24 seq=504; 15min period=96 seq=480; illness period=12 seq=504.
run_ds ETTh1       ETTh1.csv            ETTh1  7   h 24  32  504 24  96 192
run_ds ETTh2       ETTh2.csv            ETTh2  7   h 24  32  504 24  96 192
run_ds ETTm1       ETTm1.csv            ETTm1  7   t 96  32  480 96  96 192
run_ds ETTm2       ETTm2.csv            ETTm2  7   t 96  32  480 96  96 192
run_ds electricity electricity.csv      custom 321 h 168 16  504 24  96 192
run_ds weather     weather.csv          custom 21  t 144 32  504 24  96 192
run_ds traffic     traffic.csv          custom 862 h 168 8   504 24  96 192
run_ds illness     national_illness.csv custom 7   w 52  32  504 12  24 60

echo "done: ${model_name} on autoformer datasets"

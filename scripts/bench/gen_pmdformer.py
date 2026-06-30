#!/usr/bin/env python
"""Generate PMDformer runs using the official upstream best hyperparameters.

Source: aohu1105/PMDformer scripts/PMDformer/*.sh.
This tqnet benchmark root currently provides the 7 standard LTSF CSV datasets
under ./dataset.
"""

from pathlib import Path


ROOT = "./dataset"
OUT = Path("scripts/bench/commands_pmdformer.txt")
SEED = 2024
SEQ_LEN = 720
PRED_LENS = [96, 192, 336, 720]

DATASETS = {
    "ETTh1": ("ETTh1", "ETTh1.csv", 7, "h", 24),
    "ETTh2": ("ETTh2", "ETTh2.csv", 7, "h", 24),
    "ETTm1": ("ETTm1", "ETTm1.csv", 7, "t", 96),
    "ETTm2": ("ETTm2", "ETTm2.csv", 7, "t", 96),
    "electricity": ("custom", "electricity.csv", 321, "h", 168),
    "weather": ("custom", "weather.csv", 21, "t", 144),
    "traffic": ("custom", "traffic.csv", 862, "h", 168),
}


def base(ds, pred_len):
    dtype, fname, enc_in, freq, cycle = DATASETS[ds]
    return {
        "is_training": 1,
        "root_path": ROOT,
        "data_path": fname,
        "model_id": f"{ds}_{SEQ_LEN}_{pred_len}",
        "model": "PMDformer",
        "data": dtype,
        "features": "M",
        "seq_len": SEQ_LEN,
        "pred_len": pred_len,
        "enc_in": enc_in,
        "freq": freq,
        "cycle": cycle,
        "itr": 1,
        "random_seed": SEED,
    }


def hp(ds, pred_len):
    if ds == "ETTh1":
        return {
            "train_epochs": 30,
            "e_layers": 1,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.3,
            "n_heads": 4,
            "patch_size": 72,
            "d_model": 64 if pred_len == 96 else 32,
            "d_ff": 128 if pred_len == 96 else 32,
            "lradj": "type3",
            "batch_size": 128,
            "learning_rate": 0.0005,
        }
    if ds == "ETTh2":
        return {
            "train_epochs": 10,
            "e_layers": 2 if pred_len == 720 else 1,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.3,
            "n_heads": 4,
            "patch_size": 72,
            "d_model": 64,
            "d_ff": 128,
            "lradj": "type1",
            "batch_size": 32,
            "learning_rate": 0.0005,
        }
    if ds == "ETTm1":
        long_horizon = pred_len in (336, 720)
        return {
            "train_epochs": 10,
            "e_layers": 2,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.3,
            "n_heads": 4,
            "patch_size": 72,
            "d_model": 32 if long_horizon else 64,
            "d_ff": 32 if long_horizon else 64,
            "lradj": "type1",
            "batch_size": 256 if pred_len == 336 else 128,
            "learning_rate": 0.001 if pred_len == 720 else 0.01,
        }
    if ds == "ETTm2":
        short_horizon = pred_len in (96, 192)
        return {
            "train_epochs": 10,
            "e_layers": 1,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.3,
            "n_heads": 4,
            "patch_size": 48 if short_horizon else 24,
            "d_model": 64,
            "d_ff": 128,
            "lradj": "type1",
            "batch_size": 32,
            "learning_rate": 0.0005 if short_horizon else 0.0002,
        }
    if ds == "electricity":
        return {
            "train_epochs": 30,
            "e_layers": 2,
            "v_layers": 2,
            "patience": 5,
            "dropout": 0.1,
            "n_heads": 8,
            "patch_size": 72,
            "d_model": 256,
            "d_ff": 512,
            "use_norm": 0,
            "lradj": "type1",
            "batch_size": 16,
            "learning_rate": 0.0005,
        }
    if ds == "weather":
        table = {
            96: (128, 128, 0.0005),
            192: (32, 32, 0.001),
            336: (32, 64, 0.0002),
            720: (32, 32, 0.001),
        }
        d_model, d_ff, lr = table[pred_len]
        return {
            "train_epochs": 10,
            "e_layers": 2,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.1,
            "n_heads": 8,
            "patch_size": 48,
            "d_model": d_model,
            "d_ff": d_ff,
            "use_norm": 0,
            "lradj": "type1",
            "batch_size": 32,
            "learning_rate": lr,
        }
    if ds == "traffic":
        if pred_len == 192:
            heads, d_model, batch, lr = 16, 256, 16, 0.001
        elif pred_len == 96:
            heads, d_model, batch, lr = 8, 512, 32, 0.001
        else:
            heads, d_model, batch, lr = 8, 512, 16, 0.0005
        return {
            "train_epochs": 50,
            "e_layers": 3,
            "v_layers": 1,
            "patience": 5,
            "dropout": 0.1,
            "n_heads": heads,
            "patch_size": 72,
            "d_model": d_model,
            "d_ff": 512,
            "lradj": "type3",
            "batch_size": batch,
            "learning_rate": lr,
        }
    raise KeyError(ds)


def format_cmd(args):
    parts = ["python -u run.py"]
    for key, value in args.items():
        parts.append(f"--{key} {value}")
    return " ".join(parts)


def main():
    lines = []
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            args = base(ds, pred_len)
            args.update(hp(ds, pred_len))
            tag = f"PMDformer__{ds}__sl{SEQ_LEN}__pl{pred_len}"
            lines.append(f"{tag}\t{format_cmd(args)}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} runs)")


if __name__ == "__main__":
    main()

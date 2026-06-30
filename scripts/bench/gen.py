#!/usr/bin/env python
# Generate all benchmark run commands for the TQNet standard-benchmark reproduction.
#
#   12 models x 7 datasets x seq_len{96,720} x pred_len{96,192,336,720}  =  672 runs
#
# Hyperparameters are the OFFICIAL per-model github configs (see scripts/bench/HYPERPARAMS.md).
# Where a model has no official config for a given look-back, we keep the model's architectural
# hyperparameters and only change seq_len (the user-approved "缺失则合理适配").
#
# Output: scripts/bench/commands.txt  (one full `python -u run.py ...` per line, prefixed by a TAG).
#         Each line:  <TAG>\t<command>
# A driver (pbs_pair.sh) runs these 2-at-a-time on a single GPU.

import os

ROOT = "../data/Timeseries-PILE/forecasting/autoformer"
SEED = 2024
SEQ_LENS = [96, 720]
PRED_LENS = [96, 192, 336, 720]

# dataset -> (data_type, file, enc_in, freq, cycle)   cycle = CycleNet/TQNet period W
DATASETS = {
    "ETTh1":       ("ETTh1",  "ETTh1.csv",       7,   "h", 24),
    "ETTh2":       ("ETTh2",  "ETTh2.csv",       7,   "h", 24),
    "ETTm1":       ("ETTm1",  "ETTm1.csv",       7,   "t", 96),
    "ETTm2":       ("ETTm2",  "ETTm2.csv",       7,   "t", 96),
    "electricity": ("custom", "electricity.csv", 321, "h", 168),
    "weather":     ("custom", "weather.csv",     21,  "t", 144),
    "traffic":     ("custom", "traffic.csv",     862, "h", 168),
}

MODELS = ["TQNet", "iTransformer", "PatchTST", "DLinear", "NLinear", "Linear",
          "SparseTSF", "CycleNet", "SegRNN", "TimeXer", "Autoformer", "Informer"]


def base(model, ds, seq_len, pred_len):
    dtype, fname, enc_in, freq, cycle = DATASETS[ds]
    return {
        "is_training": 1,
        "root_path": ROOT,
        "data_path": fname,
        "model_id": f"{ds}_{seq_len}_{pred_len}",
        "model": model,
        "data": dtype,
        "features": "M",
        "seq_len": seq_len,
        "pred_len": pred_len,
        "enc_in": enc_in,
        "freq": freq,
        "cycle": cycle,
        "itr": 1,
        "random_seed": SEED,
    }


# ----------------------------------------------------------------------------
# Per-model hyperparameter overrides.  Each builder returns extra arg dict.
# ----------------------------------------------------------------------------

def hp_TQNet(ds, seq_len, pred_len):
    # official scripts/TQNet/*.sh : d_model=512(default), use_revin=1, model_type=mlp, lradj=type3
    tab = {  # lr, bs, dropout
        "ETTh1": (0.001, 256, 0.5), "ETTh2": (0.001, 256, 0.5),
        "ETTm1": (0.001, 256, 0.5), "ETTm2": (0.001, 256, 0.5),
        "electricity": (0.003, 32, 0.0), "weather": (0.001, 64, 0.5),
        "traffic": (0.003, 16, 0.0),
    }
    lr, bs, dp = tab[ds]
    return {"learning_rate": lr, "batch_size": bs, "dropout": dp,
            "train_epochs": 30, "patience": 5, "lradj": "type3"}


def hp_CycleNet(ds, seq_len, pred_len):
    # CycleNet MLP variant (headline). model_type=mlp d_model=512 use_revin=1 lradj=type3
    tab = {  # lr, bs
        "ETTh1": (0.005, 256), "ETTh2": (0.005, 256),
        "ETTm1": (0.005, 256), "ETTm2": (0.005, 256),
        "electricity": (0.005, 64), "weather": (0.005, 256),
        "traffic": (0.002, 64),
    }
    lr, bs = tab[ds]
    return {"model_type": "mlp", "d_model": 512, "use_revin": 1,
            "learning_rate": lr, "batch_size": bs,
            "train_epochs": 30, "patience": 5, "lradj": "type3"}


def hp_iTransformer(ds, seq_len, pred_len):
    # official iTransformer scripts (seq_len=96). n_heads=8 dropout=0.1 factor=1 epochs=10
    if ds == "ETTh1":
        dmodel = 256 if pred_len in (96, 192) else 512
        e_layers, lr, bs = 2, 1e-4, 32
    elif ds == "ETTh2":
        dmodel, e_layers, lr, bs = 128, 2, 1e-4, 32
    elif ds in ("ETTm1", "ETTm2"):
        dmodel, e_layers, lr, bs = 128, 2, 1e-4, 32
    elif ds == "electricity":
        dmodel, e_layers, lr, bs = 512, 3, 5e-4, 16
    elif ds == "weather":
        dmodel, e_layers, lr, bs = 512, 3, 1e-4, 32
    elif ds == "traffic":
        dmodel, e_layers, lr, bs = 512, 4, 1e-3, 16
    return {"e_layers": e_layers, "d_model": dmodel, "d_ff": dmodel, "n_heads": 8,
            "dropout": 0.1, "factor": 1, "use_revin": 1, "learning_rate": lr,
            "batch_size": bs, "train_epochs": 10, "patience": 3, "lradj": "type1"}


def hp_PatchTST(ds, seq_len, pred_len):
    # official PatchTST supervised (seq_len=336). patch_len=16 stride=8 e_layers=3 revin=1
    common = {"e_layers": 3, "patch_len": 16, "stride": 8, "revin": 1,
              "individual": 0, "decomposition": 0, "head_dropout": 0.0,
              "learning_rate": 1e-4, "train_epochs": 100}
    if ds in ("ETTh1", "ETTh2"):
        common.update({"n_heads": 4, "d_model": 16, "d_ff": 128, "dropout": 0.3,
                       "fc_dropout": 0.3, "batch_size": 128, "patience": 100,
                       "lradj": "type3", "pct_start": 0.3})
    elif ds in ("ETTm1", "ETTm2"):
        common.update({"n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
                       "fc_dropout": 0.2, "batch_size": 128, "patience": 20,
                       "lradj": "TST", "pct_start": 0.4})
    elif ds == "electricity":
        common.update({"n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
                       "fc_dropout": 0.2, "batch_size": 32, "patience": 10,
                       "lradj": "TST", "pct_start": 0.2})
    elif ds == "weather":
        common.update({"n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
                       "fc_dropout": 0.2, "batch_size": 128, "patience": 20,
                       "lradj": "type3", "pct_start": 0.3})
    elif ds == "traffic":
        # official bs=24 at sl336; at sl720 (862 ch) that OOMs even solo on a 140GB H200,
        # so halve to 12 for the long look-back (run with MAXPROC=1). Adaptation, not official.
        bs = 12 if seq_len == 720 else 24
        common.update({"n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
                       "fc_dropout": 0.2, "batch_size": bs, "patience": 10,
                       "lradj": "TST", "pct_start": 0.2})
    return common


def hp_Linear_family(ds, seq_len, pred_len):
    # LTSF-Linear (DLinear/NLinear/Linear), seq_len=336 official. individual=0 lradj=type1
    tab = {  # lr, bs   (ETTm2 lr is per-pred-len, handled below; weather lr=default 1e-4)
        "ETTh1": (0.005, 32), "ETTh2": (0.05, 32),
        "ETTm1": (0.0001, 8), "ETTm2": (None, 32),
        "electricity": (0.001, 16), "weather": (0.0001, 16),
        "traffic": (0.05, 16),
    }
    lr, bs = tab[ds]
    if ds == "ETTm2":
        lr = 0.001 if pred_len in (96, 192) else 0.01
    return {"individual": 0, "learning_rate": lr, "batch_size": bs,
            "train_epochs": 10, "patience": 3, "lradj": "type1"}


def hp_SparseTSF(ds, seq_len, pred_len):
    # official SparseTSF linear variant (seq_len=720). MUST pass model_type=linear (repo default mlp).
    period = 4 if ds in ("ETTm1", "ETTm2", "weather") else 24
    tab = {  # lr, bs
        "ETTh1": (0.02, 256), "ETTh2": (0.03, 256),
        "ETTm1": (0.02, 256), "ETTm2": (0.02, 256),
        "electricity": (0.02, 128), "weather": (0.02, 256),
        "traffic": (0.03, 128),
    }
    lr, bs = tab[ds]
    return {"model_type": "linear", "period_len": period,
            "learning_rate": lr, "batch_size": bs, "train_epochs": 30,
            "patience": 5, "lradj": "type3"}


def hp_SegRNN(ds, seq_len, pred_len):
    # SegRNN: Lookback_96 config for seq_len=96, main(720) config for seq_len=720.
    # gru/pmf, d_model=512. revin forced 1 (SegRNN's own default is on; repo default is 0).
    if seq_len == 96:
        tab = {  # seg_len, dropout, lr, bs, channel_id
            "ETTh1": (24, 0.5, 0.001, 256, 1), "ETTh2": (24, 0.5, 0.0002, 256, 1),
            "ETTm1": (48, 0.5, 0.0002, 256, 1), "ETTm2": (48, 0.5, 0.0001, 256, 1),
            "electricity": (24, 0.0, 0.0005, 16, 1), "weather": (24, 0.5, 0.0001, 64, 1),
            "traffic": (24, 0.0, 0.003, 8, 0),
        }
        patience = 10
    else:  # seq_len == 720
        tab = {
            "ETTh1": (24, 0.1, 0.0003, 64, 1), "ETTh2": (24, 0.5, 0.0003, 64, 0),
            "ETTm1": (48, 0.5, 0.0003, 256, 1), "ETTm2": (48, 0.5, 0.0003, 256, 0),
            "electricity": (48, 0.1, 0.0003, 32, 1), "weather": (48, 0.1, 0.0001, 128, 1),
            "traffic": (48, 0.1, 0.001, 8, 0),
        }
        patience = 5
    seg, dp, lr, bs, cid = tab[ds]
    return {"rnn_type": "gru", "dec_way": "pmf", "seg_len": seg, "d_model": 512,
            "dropout": dp, "channel_id": cid, "revin": 1, "learning_rate": lr,
            "batch_size": bs, "train_epochs": 30, "patience": patience}


def hp_TimeXer(ds, seq_len, pred_len):
    # official TimeXer multivariate scripts (seq_len=96, label_len=48, factor=3, patch_len=16,
    # n_heads=8, dropout=0.1). Heavy per-pred-len variation in (e_layers,d_model,d_ff,batch).
    T = {
        "ETTh1": {96: (1, 256, 2048, 4), 192: (2, 128, 2048, 4), 336: (1, 512, 1024, 16), 720: (1, 256, 1024, 16)},
        "ETTh2": {96: (1, 256, 1024, 16), 192: (1, 256, 1024, 32), 336: (2, 512, 1024, 4), 720: (2, 256, 1024, 16)},
        "ETTm1": {96: (1, 256, 2048, 4), 192: (1, 256, 256, 4), 336: (1, 256, 1024, 4), 720: (1, 256, 512, 4)},
        "ETTm2": {96: (1, 256, 2048, 32), 192: (1, 256, 1024, 16), 336: (1, 512, 1024, 32), 720: (1, 512, 2048, 32)},
        "electricity": {96: (4, 512, 512, 4), 192: (3, 512, 2048, 4), 336: (4, 512, 2048, 4), 720: (3, 512, 2048, 4)},
        "weather": {96: (1, 256, 512, 4), 192: (3, 128, 1024, 4), 336: (1, 256, 2048, 4), 720: (1, 128, 2048, 4)},
        "traffic": {96: (3, 512, 512, 16), 192: (3, 512, 512, 16), 336: (2, 512, 512, 16), 720: (2, 512, 512, 16)},
    }
    e_layers, d_model, d_ff, bs = T[ds][pred_len]
    lr = 1e-3 if ds == "traffic" else 1e-4
    return {"label_len": 48, "factor": 3, "patch_len": 16, "n_heads": 8,
            "e_layers": e_layers, "d_model": d_model, "d_ff": d_ff, "dropout": 0.1,
            "use_revin": 1, "learning_rate": lr, "batch_size": bs,
            "train_epochs": 10, "patience": 3, "lradj": "type1"}


def hp_Autoformer(ds, seq_len, pred_len):
    enc_in = DATASETS[ds][2]
    factor = 1 if ds == "ETTm2" else 3
    epochs = {"weather": 2, "traffic": 3}.get(ds, 10)
    return {"dec_in": enc_in, "c_out": enc_in,
            "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": factor,
            "d_model": 512, "d_ff": 2048, "n_heads": 8, "moving_avg": 25,
            "dropout": 0.05, "embed": "timeF", "learning_rate": 1e-4, "batch_size": 32,
            "train_epochs": epochs, "patience": 3, "lradj": "type1"}


def hp_Informer(ds, seq_len, pred_len):
    enc_in = DATASETS[ds][2]
    return {"dec_in": enc_in, "c_out": enc_in,
            "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5,
            "d_model": 512, "d_ff": 2048, "n_heads": 8, "dropout": 0.05,
            "embed": "timeF", "learning_rate": 1e-4, "batch_size": 32,
            "train_epochs": 10, "patience": 3, "lradj": "type1"}


def hp_TimesNet(ds, seq_len, pred_len):
    # TSLib official (seq96). e_layers=2, top_k=5, num_kernels=6, factor=3, dropout=0.1, lr=1e-4, bs=32.
    enc_in = DATASETS[ds][2]
    T = {  # (d_model, d_ff, epochs) per pred_len
        "ETTh1": {p: (16, 32, 10) for p in PRED_LENS},
        "ETTh2": {p: (32, 32, 10) for p in PRED_LENS},
        "ETTm1": {96: (64, 64, 10), 192: (64, 64, 10), 336: (16, 32, 3), 720: (16, 32, 10)},
        "ETTm2": {96: (32, 32, 10), 192: (32, 32, 1), 336: (32, 32, 10), 720: (16, 32, 1)},
        "electricity": {p: (256, 512, 10) for p in PRED_LENS},
        "weather": {96: (32, 32, 10), 192: (32, 32, 1), 336: (32, 32, 10), 720: (32, 32, 1)},
        "traffic": {p: (512, 512, 10) for p in PRED_LENS},
    }
    d_model, d_ff, epochs = T[ds][pred_len]
    return {"c_out": enc_in, "label_len": 48, "e_layers": 2, "d_model": d_model, "d_ff": d_ff,
            "top_k": 5, "num_kernels": 6, "factor": 3, "dropout": 0.1, "learning_rate": 1e-4,
            "batch_size": 32, "train_epochs": epochs, "patience": 3, "lradj": "type1"}


def hp_Crossformer(ds, seq_len, pred_len):
    # TSLib official (seq96, label48; traffic label96). e_layers=2, d_layers=1, dropout=0.1, lr=1e-4.
    enc_in = DATASETS[ds][2]
    # (d_model, d_ff, n_heads, factor, batch, label_len)
    C = {
        "ETTh1": (512, 2048, 8, 3, 32, 48), "ETTh2": (512, 2048, 8, 3, 32, 48),
        "ETTm1": (512, 2048, 8, 1, 32, 48), "ETTm2": (512, 2048, 8, 1, 32, 48),
        "electricity": (256, 512, 8, 3, 16, 48), "weather": (32, 32, 8, 3, 32, 48),
        "traffic": (512, 2048, 2, 3, 4, 96),
    }
    d_model, d_ff, n_heads, factor, bs, ll = C[ds]
    epochs = 1 if (ds == "weather" and pred_len in (192, 720)) else 10
    return {"c_out": enc_in, "label_len": ll, "e_layers": 2, "d_layers": 1,
            "d_model": d_model, "d_ff": d_ff, "n_heads": n_heads, "factor": factor,
            "dropout": 0.1, "learning_rate": 1e-4, "batch_size": bs,
            "train_epochs": epochs, "patience": 3, "lradj": "type1"}


def hp_MSGNet(ds, seq_len, pred_len):
    # MSGNet official (YoZhibo/MSGNet, seq96). subgraph_size=3, gcn_depth=2, propalpha=0.3,
    # skip_channel=32, n_heads=8, bs=32, lr=1e-4. traffic has NO official config -> adapted (noted).
    enc_in = DATASETS[ds][2]
    # (e_layers, d_model, d_ff, top_k, conv_channel, node_dim, dropout, epochs)
    if ds == "ETTh1":
        M = {96: (1, 32, 64, 3, 32, 10, 0.1, 10), 192: (1, 32, 64, 3, 32, 10, 0.1, 10),
             336: (2, 32, 64, 3, 32, 10, 0.1, 10), 720: (1, 16, 32, 3, 32, 10, 0.1, 10)}[pred_len]
    elif ds == "ETTh2":
        M = (2, 16, 32, 5, 32, 10, 0.05, 10)
    elif ds == "ETTm1":
        conv = 32 if pred_len == 96 else 16
        M = (1, 32, 32, 3, conv, 10, 0.05, 10)
    elif ds == "ETTm2":
        d_ff = 64 if pred_len in (192, 720) else 32
        M = (2, 32, d_ff, 3, 32, 10, 0.3, 10)
    elif ds == "electricity":
        e = 3 if pred_len in (336, 720) else 2
        M = (e, 1024, 512, 5, 16, 100, 0.05, 10)
    elif ds == "weather":
        e = 1 if pred_len == 336 else 2
        ep = 3 if pred_len == 96 else 10
        M = (e, 64, 128, 5, 32, 10, 0.05, ep)
    elif ds == "traffic":
        # no official MSGNet traffic config -> adapted. MSGNet's GraphBlock needs d_model >= enc_in
        # (start_conv kernel = d_model - c_out + 1), so 862-channel traffic needs d_model >= 862.
        M = (2, 1024, 512, 5, 16, 100, 0.05, 10)
    e_layers, d_model, d_ff, top_k, conv, node_dim, dropout, epochs = M
    bs = 4 if ds == "traffic" else 32
    return {"c_out": enc_in, "label_len": 48, "e_layers": e_layers, "d_model": d_model,
            "d_ff": d_ff, "top_k": top_k, "conv_channel": conv, "skip_channel": 32,
            "node_dim": node_dim, "subgraph_size": 3, "gcn_depth": 2, "propalpha": 0.3,
            "n_heads": 8, "individual": 0, "dropout": dropout, "learning_rate": 1e-4,
            "batch_size": bs, "train_epochs": epochs, "patience": 3, "lradj": "type1"}


def hp_SCINet(ds, seq_len, pred_len):
    # No official tuned config exists (TSLib has the model but no script; cure-lab uses a
    # different/incomparable protocol). Run with TSLib generic defaults, 2 stacks (d_layers=2).
    # SCINet hardcodes levels=3 (needs seq_len % 8 == 0: 96,720 OK) and kernel=5.
    enc_in = DATASETS[ds][2]
    bs = 8 if ds == "traffic" else (16 if ds == "electricity" else 32)
    return {"c_out": enc_in, "label_len": 48, "d_layers": 2, "dropout": 0.1,
            "learning_rate": 1e-4, "batch_size": bs, "train_epochs": 10,
            "patience": 3, "lradj": "type1"}


BUILDERS = {
    "TQNet": hp_TQNet, "CycleNet": hp_CycleNet, "iTransformer": hp_iTransformer,
    "PatchTST": hp_PatchTST, "DLinear": hp_Linear_family, "NLinear": hp_Linear_family,
    "Linear": hp_Linear_family, "SparseTSF": hp_SparseTSF, "SegRNN": hp_SegRNN,
    "TimeXer": hp_TimeXer, "Autoformer": hp_Autoformer, "Informer": hp_Informer,
    "TimesNet": hp_TimesNet, "Crossformer": hp_Crossformer, "MSGNet": hp_MSGNet, "SCINet": hp_SCINet,
}


def build_cmd(model, ds, seq_len, pred_len):
    args = base(model, ds, seq_len, pred_len)
    args.update(BUILDERS[model](ds, seq_len, pred_len))
    parts = ["python", "-u", "run.py"]
    for k, v in args.items():
        parts.append(f"--{k}")
        parts.append(str(v))
    return " ".join(parts)


def main():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    # optional model filter: `python gen.py TimesNet Crossformer ...` -> commands_new.txt
    sel = sys.argv[1:]
    if sel:
        models = sel
        out = os.path.join(here, "commands_new.txt")
    else:
        models = MODELS
        out = os.path.join(here, "commands.txt")
    n = 0
    with open(out, "w") as f:
        for model in models:
            for ds in DATASETS:
                for sl in SEQ_LENS:
                    for pl in PRED_LENS:
                        tag = f"{model}__{ds}__sl{sl}__pl{pl}"
                        cmd = build_cmd(model, ds, sl, pl)
                        f.write(f"{tag}\t{cmd}\n")
                        n += 1
    print(f"wrote {n} commands -> {out}")


if __name__ == "__main__":
    main()

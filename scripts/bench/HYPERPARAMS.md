# Benchmark hyperparameters — sources & values

Reproduction of the TQNet paper (https://arxiv.org/pdf/2505.12917) standard benchmark.
12 models × 7 datasets × seq_len∈{96,720} × pred_len∈{96,192,336,720} = **672 runs**.
All values are the **official per-model GitHub configs**. Where a model has no official
config for a given look-back, its architectural hyperparameters are kept and only `seq_len`
changes (user-approved "缺失则合理适配"). Encoded in `gen.py`.

Common to all: `features=M`, `seed=2024`, `itr=1`, metrics in scaled/z-score space (no
inverse_transform — standard LTSF protocol). `enc_in`: ETT=7, electricity=321, weather=21,
traffic=862. `cycle` (W, used by CycleNet/TQNet): ETTh=24, ETTm=96, ECL/traffic=168, weather=144.

> Note on look-back: each model family has a *native* look-back in its official scripts
> (formers/iTransformer/CycleNet/TQNet → 96; PatchTST & LTSF-Linear → 336; SparseTSF & SegRNN
> → 720). We run every model at **both** 96 and 720. At the non-native look-back we reuse the
> model's published architectural HPs; SegRNN additionally has an official Lookback_96 config
> which we use for seq_len=96.

| Model | Source repo | Key hyperparameters |
|---|---|---|
| **TQNet** | ACAT-SCUT/TQNet `scripts/TQNet/*.sh` | d_model=512, model_type=mlp, use_revin=1, lradj=type3, ep=30, pat=5. dropout=0.5 (ETT/weather) / 0 (ECL,traffic). lr/bs: ETT 0.001/256, weather 0.001/64, ECL 0.003/32, traffic 0.003/16. |
| **CycleNet** | ACAT-SCUT/CycleNet `MLP-Input-96/*.sh` | MLP variant (headline). d_model=512, use_revin=1, lradj=type3, ep=30, pat=5. lr=0.005 (traffic 0.002). bs: ETT/weather 256, ECL/traffic 64. |
| **iTransformer** | thuml/iTransformer `multivariate_forecasting/*` | n_heads=8, dropout=0.1, factor=1, lradj=type1, ep=10, pat=3. e_layers/d_model: ETTh1 2/256→512(pl≥336), ETTh2 2/128, ETTm 2/128, ECL 3/512 (lr5e-4,bs16), weather 3/512, traffic 4/512 (lr1e-3,bs16). |
| **PatchTST** | yuqinie98/PatchTST (supervised) | patch=16, stride=8, e_layers=3, revin=1, lr=1e-4, ep=100. ETTh: heads4 d16 d_ff128 dp0.3 bs128 pat100 type3. ETTm/weather: heads16 d128 d_ff256 dp0.2 bs128 pat20. ECL: bs32 pat10 TST. traffic: bs24 pat10 TST. |
| **DLinear/NLinear/Linear** | cure-lab/LTSF-Linear | individual=0, lradj=type1, ep=10, pat=3. lr/bs: ETTh1 0.005/32, ETTh2 0.05/32, ETTm1 1e-4/8, ETTm2 (1e-3 pl≤192 / 1e-2 pl≥336)/32, ECL 1e-3/16, weather 1e-4/16, traffic 0.05/16. |
| **SparseTSF** | lss-1138/SparseTSF `linear/*.sh` | model_type=linear (explicit; repo default is mlp), ep=30, pat=5. period_len=24 (hourly: ETTh,ECL,traffic) / 4 (ETTm,weather). lr/bs: ETTh1 0.02/256, ETTh2 0.03/256, ETTm 0.02/256, ECL 0.02/128, weather 0.02/256, traffic 0.03/128. |
| **SegRNN** | lss-1138/SegRNN (`*` + `Lookback_96/*`) | gru, pmf, d_model=512, revin=1 (forced; SegRNN default on). **sl96** (Lookback_96): seg=24 (ETTm 48), pat=10; lr/bs/dp/channel_id per dataset. **sl720** (main): seg=24 (ETTh)/48 (rest), pat=5. |
| **TimeXer** | thuml/TimeXer `multivariate/*` | label_len=48, factor=3, patch=16, n_heads=8, dropout=0.1, lradj=type1, ep=10, pat=3. e_layers/d_model/d_ff/bs vary per (dataset,pred_len) — exact table in `gen.py` (hp_TimeXer). traffic lr=1e-3 else 1e-4. |
| **Autoformer** | thuml/Autoformer (TSLib scripts) | label_len=48, e_layers=2, d_layers=1, d_model=512, d_ff=2048, n_heads=8, moving_avg=25, dropout=0.05, factor=3 (ETTm2=1), lr=1e-4, bs=32, lradj=type1, pat=3. ep=10 (weather 2, traffic 3). |
| **Informer** | zhouhaoyi/Informer2020 / TSLib | label_len=48, e_layers=2, d_layers=1, d_model=512, d_ff=2048, n_heads=8, dropout=0.05, factor=5, lr=1e-4, bs=32, lradj=type1, ep=10, pat=3. |
| **PMDformer** | aohu1105/PMDformer `scripts/PMDformer/*.sh` | Official-best seq_len=720 configs for ETTh1/2, ETTm1/2, ECL, weather, traffic. Exact per-(dataset,horizon) settings are encoded in `gen_pmdformer.py`: patch_size 72 except ETTm2/weather variants, PMDformer `v_layers`, `use_norm`, d_model/d_ff, lr/bs/lradj copied from upstream scripts. |

## Added baselines (TQNet paper Table-5), ported from thuml/Time-Series-Library

| Model | Source | Key hyperparameters |
|---|---|---|
| **TimesNet** | TSLib `long_term_forecast` scripts | seq96, label48, e_layers=2, top_k=5, num_kernels=6, factor=3, dropout=0.1, lr=1e-4, bs=32. d_model/d_ff per (ds,H): ETTh1 16/32, ETTh2 32/32, ETTm1 64/64 (336/720→16/32), ETTm2 32/32 (720→16/32), ECL 256/512, weather 32/32, traffic 512/512. Some ETTm/weather H use train_epochs 1/3 (TSLib, model is slow). |
| **Crossformer** | TSLib `long_term_forecast` scripts | seq96, label48 (**traffic label96**), e_layers=2, d_layers=1, dropout=0.1, lr=1e-4. d_model/d_ff: ETT 512/2048, ECL 256/512 (bs16), weather 32/32, traffic 512/2048 (n_heads=2, bs4). factor=3 (ETTm1/m2=1). n_heads=8 except traffic=2. |
| **MSGNet** | YoZhibo/MSGNet scripts | seq96, label48, subgraph_size=3, gcn_depth=2, propalpha=0.3, skip_channel=32, n_heads=8, bs=32, lr=1e-4. Per (ds,H): e_layers/d_model/d_ff/top_k/conv_channel/node_dim/dropout from official scripts (ECL d_model=1024, node_dim=100). **traffic: NO official config → adapted** (e_layers2, d_model256, d_ff512, conv16, node_dim100, bs8). |
| **SCINet** | TSLib model (no tuned script anywhere) | **No official tuned config exists** for the standard seq96/pred{96..720} protocol (TSLib ships the model but no script; cure-lab uses an incompatible protocol). Run with TSLib generic defaults: label48, d_layers=2 (stacks), dropout=0.1, lr=1e-4, bs=32 (ECL16/traffic8), ep10. SCINet hardcodes levels=3 (needs seq%8==0; 96,720 OK) & kernel=5. **Flagged as untuned.** |

Notes for the 4 added models: `c_out` set to `enc_in` (decoder output channels); they route to the former-style 4-arg call in `exp_main`; heavy ones (TimesNet/MSGNet on traffic & electricity) submit with MAXPROC=1 to avoid OOM.

**Repo-specific gotchas handled in `gen.py`:**
- `--model_type linear` set explicitly for SparseTSF (this repo defaults to `mlp`).
- `--revin 1` set explicitly for SegRNN (this repo's default is 0; SegRNN's own default is on).
- `--label_len 48` only for Autoformer/Informer/TimeXer (others ignore it; `exp_main` always uses MSELoss, so `--loss` is irrelevant).
- `period_len` (SparseTSF) and `seg_len` (SegRNN) both divide all of {96,192,336,720} and the look-back, so divisibility holds at both seq_len=96 and 720.

#!/usr/bin/env python
# Build results/BENCHMARK_RESULTS.xlsx (paper Table-2 layout) from logs/bench/*.log.
# - one sheet per look-back seq_len (+ an Overall-avg sheet)
# - models as columns (MSE | MAE pairs, model name merged across the pair)
# - dataset x horizon as rows, per-dataset Avg row
# - bold = best in row, underline = 2nd-best (per metric, per row)
# - frozen header (top 2 rows) AND frozen Dataset/H columns (left 2 cols) for easy scrolling

import os
import re
import glob
import xlsxwriter

HERE = os.path.dirname(os.path.abspath(__file__))
TQNET = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGDIR = os.path.join(TQNET, "logs", "bench")
OUT = os.path.join(TQNET, "results", "BENCHMARK_RESULTS.xlsx")

MODELS = ["TQNet", "TimeXer", "CycleNet", "iTransformer", "MSGNet", "TimesNet", "PatchTST",
          "Crossformer", "DLinear", "SCINet", "NLinear", "Linear", "SparseTSF", "SegRNN",
          "Autoformer", "Informer"]
MODEL_HEAD = {"TQNet": "TQNet (Ours)"}
DATASETS = [("ETTh1", "ETTh1"), ("ETTh2", "ETTh2"), ("ETTm1", "ETTm1"), ("ETTm2", "ETTm2"),
            ("Electricity", "electricity"), ("Solar", "Solar"), ("Traffic", "traffic"),
            ("Weather", "weather"), ("PEMS03", "PEMS03"), ("PEMS04", "PEMS04"),
            ("PEMS07", "PEMS07"), ("PEMS08", "PEMS08")]
SEQ_LENS = [96, 720]
STANDARD_PRED_LENS = [96, 192, 336, 720]
PEMS_PRED_LENS = [12, 24, 48, 96]
# intentionally-not-run cells -> rendered "N/A" (MSGNet/traffic/sl720: infeasible, no official cfg)
NA = {("MSGNet", "traffic", 720, p) for p in STANDARD_PRED_LENS}


def pred_lens_for(ds):
    return PEMS_PRED_LENS if ds.startswith("PEMS") else STANDARD_PRED_LENS

MSE_RE = re.compile(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)")
TAG_RE = re.compile(r"^(?P<model>[A-Za-z]+)__(?P<ds>[A-Za-z0-9]+)__sl(?P<sl>\d+)__pl(?P<pl>\d+)$")


def parse_logs():
    res = {}
    for log in glob.glob(os.path.join(LOGDIR, "*.log")):
        tag = os.path.basename(log)[:-4]
        m = TAG_RE.match(tag)
        if not m:
            continue
        mse = mae = None
        with open(log, errors="ignore") as f:
            for line in f:
                mm = MSE_RE.search(line)
                if mm:
                    mse, mae = float(mm.group(1)), float(mm.group(2))
        if mse is not None:
            res[(m["model"], m["ds"], int(m["sl"]), int(m["pl"]))] = (mse, mae)
    return res


def rank_marks(values):
    vis = sorted([(v, i) for i, v in enumerate(values) if v is not None])
    marks = {}
    if vis:
        marks[vis[0][1]] = "best"
        for v, i in vis[1:]:
            if v > vis[0][0] + 1e-12:
                marks[i] = "second"
                break
    return marks


def main():
    res = parse_logs()
    done = len(res)
    expected = [(m, key, sl, pl)
                for m in MODELS for _, key in DATASETS for sl in SEQ_LENS
                for pl in pred_lens_for(key)]
    total = len(expected) - len(NA)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb = xlsxwriter.Workbook(OUT)

    # ---- formats ----
    f_title = wb.add_format({"bold": True, "font_size": 12})
    f_note = wb.add_format({"italic": True, "font_color": "#555555", "text_wrap": True})
    f_mhead = wb.add_format({"bold": True, "align": "center", "valign": "vcenter",
                             "bg_color": "#1F3864", "font_color": "white", "border": 1})
    f_mhead_ours = wb.add_format({"bold": True, "align": "center", "valign": "vcenter",
                                  "bg_color": "#C00000", "font_color": "white", "border": 1})
    f_sub = wb.add_format({"bold": True, "align": "center", "bg_color": "#D9E1F2", "border": 1})
    f_corner = wb.add_format({"bold": True, "align": "center", "valign": "vcenter",
                              "bg_color": "#D9E1F2", "border": 1})
    f_ds = wb.add_format({"bold": True, "align": "center", "valign": "vcenter",
                          "bg_color": "#F2F2F2", "border": 1})
    f_h = wb.add_format({"align": "center", "border": 1})
    f_h_avg = wb.add_format({"align": "center", "border": 1, "bold": True, "bg_color": "#FFF2CC"})

    def numfmt(mark, avg):
        d = {"num_format": "0.000", "align": "center", "border": 1}
        if avg:
            d["bg_color"] = "#FFF2CC"
        if mark == "best":
            d["bold"] = True
            d["font_color"] = "#006100"
        elif mark == "second":
            d["underline"] = True
            d["font_color"] = "#9C5700"
        return wb.add_format(d)
    fmt_cache = {}

    def cellfmt(mark, avg):
        k = (mark, avg)
        if k not in fmt_cache:
            fmt_cache[k] = numfmt(mark, avg)
        return fmt_cache[k]

    ncol = 2 + 2 * len(MODELS)

    def build_sheet(ws, sl):
        ws.set_column(0, 0, 12)   # Dataset
        ws.set_column(1, 1, 6)    # H
        ws.set_column(2, ncol - 1, 9)
        # header row 0: model names merged across the MSE/MAE pair
        ws.merge_range(0, 0, 1, 0, "Dataset", f_corner)
        ws.merge_range(0, 1, 1, 1, "H", f_corner)
        for j, m in enumerate(MODELS):
            c0 = 2 + 2 * j
            fmt = f_mhead_ours if m == "TQNet" else f_mhead
            ws.merge_range(0, c0, 0, c0 + 1, MODEL_HEAD.get(m, m), fmt)
            ws.write(1, c0, "MSE", f_sub)
            ws.write(1, c0 + 1, "MAE", f_sub)
        ws.freeze_panes(2, 2)

        r = 2
        for disp, key in DATASETS:
            block_start = r
            sums = {m: [0.0, 0.0, 0] for m in MODELS}
            pred_lens = pred_lens_for(key)
            for pl in pred_lens:
                mses = [(res.get((m, key, sl, pl)) or (None, None))[0] for m in MODELS]
                maes = [(res.get((m, key, sl, pl)) or (None, None))[1] for m in MODELS]
                mk_mse = rank_marks(mses)
                mk_mae = rank_marks(maes)
                ws.write(r, 1, pl, f_h)
                for j, m in enumerate(MODELS):
                    c0 = 2 + 2 * j
                    v_mse, v_mae = mses[j], maes[j]
                    if (m, key, sl, pl) in NA:
                        ws.write(r, c0, "N/A", cellfmt(None, False)); ws.write(r, c0 + 1, "N/A", cellfmt(None, False))
                        continue
                    ws.write(r, c0, v_mse if v_mse is not None else "—", cellfmt(mk_mse.get(j), False))
                    ws.write(r, c0 + 1, v_mae if v_mae is not None else "—", cellfmt(mk_mae.get(j), False))
                    if v_mse is not None:
                        sums[m][0] += v_mse; sums[m][1] += v_mae; sums[m][2] += 1
                r += 1
            # Avg row
            avg_mse = [(sums[m][0] / sums[m][2]) if sums[m][2] == len(pred_lens) else None for m in MODELS]
            avg_mae = [(sums[m][1] / sums[m][2]) if sums[m][2] == len(pred_lens) else None for m in MODELS]
            mk_mse = rank_marks(avg_mse); mk_mae = rank_marks(avg_mae)
            ws.write(r, 1, "Avg", f_h_avg)
            for j, m in enumerate(MODELS):
                c0 = 2 + 2 * j
                if (m, key, sl, pred_lens[0]) in NA:
                    ws.write(r, c0, "N/A", cellfmt(None, True)); ws.write(r, c0 + 1, "N/A", cellfmt(None, True))
                    continue
                ws.write(r, c0, avg_mse[j] if avg_mse[j] is not None else "—", cellfmt(mk_mse.get(j), True))
                ws.write(r, c0 + 1, avg_mae[j] if avg_mae[j] is not None else "—", cellfmt(mk_mae.get(j), True))
            r += 1
            # merge dataset name down the block
            ws.merge_range(block_start, 0, r - 1, 0, disp, f_ds)

    for sl in SEQ_LENS:
        build_sheet(wb.add_worksheet(f"seq_len={sl}"), sl)

    # overall-avg sheet
    ws = wb.add_worksheet("Overall avg")
    ws.set_column(0, 0, 16); ws.set_column(1, 4, 14)
    ws.write(0, 0, "Model", f_corner)
    for j, lab in enumerate(["avg MSE (sl96)", "avg MAE (sl96)", "avg MSE (sl720)", "avg MAE (sl720)"]):
        ws.write(0, 1 + j, lab, f_sub)
    ws.freeze_panes(1, 1)
    rr = 1
    for m in MODELS:
        ws.write(rr, 0, MODEL_HEAD.get(m, m), f_ds)
        col = 1
        for sl in SEQ_LENS:
            for mi in (0, 1):
                vals = [(res.get((m, key, sl, pl)) or (None, None))[mi]
                        for _, key in DATASETS for pl in pred_lens_for(key)]
                vals = [v for v in vals if v is not None]
                ws.write(rr, col, (sum(vals) / len(vals)) if vals else "—",
                         wb.add_format({"num_format": "0.000", "align": "center", "border": 1}))
                col += 1
        rr += 1

    # info sheet
    ws = wb.add_worksheet("info")
    ws.set_column(0, 0, 100)
    ws.write(0, 0, "TQNet Standard-Benchmark Reproduction", f_title)
    for i, line in enumerate([
        "Paper: Temporal Query Network for Efficient MTS Forecasting (ICML 2025) — arxiv.org/abs/2505.12917",
        f"Runs complete: {done} / {total}.",
        "Metric = MSE / MAE in standard scaled (z-score) space; lower is better.",
        "Bold green = best in row; underlined orange = 2nd-best (per metric). Avg = mean over 4 horizons.",
        "12 datasets from TQNet Table 2. Non-PEMS horizon H in {96,192,336,720}; PEMS horizon H in {12,24,48,96}.",
        "Hyperparameters: official per-model GitHub configs (see scripts/bench/HYPERPARAMS.md).",
        "Header rows + Dataset/H columns are frozen for scrolling.",
    ], start=2):
        ws.write(i, 0, line, f_note)

    wb.close()
    print(f"wrote {OUT}  ({done}/{total} runs)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Regression modeling.

Reads a modeling CSV (imaging + clinical + target, with a 0/1 ``split`` column)
and, optionally, an external-test CSV of the same schema. Uses a single feature
set (imaging + the clinical columns in config), trains the model panel
(LinearRegression / LASSO / ElasticNet / SVR / RandomForest / XGBoost /
LightGBM [+ TabPFN]) on ``split == 0``, and reports metrics (RMSE/MAE/MBE/MAPE,
original scale) on ``split == 1`` and the external test, plus
RepeatedStratifiedKFold(5,3) CV RMSE.

Usage:
    python scripts/06_run_modeling.py
    python scripts/06_run_modeling.py --mode hpo --n-trials 50
    python scripts/06_run_modeling.py --train data/train_val.csv --test data/external_test.csv
"""

from __future__ import annotations

import argparse
import os

from _common import load_config, resolve


def main():
    cfg = load_config()
    mcfg = cfg["modeling"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=None, help="train/val CSV (needs 'split' 0/1)")
    p.add_argument("--test", default=None, help="external test CSV (optional)")
    p.add_argument("--results", default=None, help="output dir")
    p.add_argument("--mode", default=mcfg.get("mode", "default"), choices=["default", "hpo"])
    p.add_argument("--n-trials", type=int, default=mcfg.get("n_trials", 30))
    p.add_argument("--models", default=None,
                   help="comma-separated subset of models (default: config / all)")
    p.add_argument("--tabpfn-phe", action="store_true",
                   help="also fit a TabPFN post-hoc ensemble (needs tabpfn_extensions)")
    args = p.parse_args()

    import pandas as pd

    from kidney_radiomics import modeling as M

    if args.n_trials is not None:
        mcfg = {**mcfg, "n_trials": args.n_trials}

    results_dir = resolve(args.results or cfg["output"]["dir"])
    os.makedirs(results_dir, exist_ok=True)
    train_path = resolve(args.train or cfg["data"]["clinical_csv"])
    test_arg = args.test or mcfg.get("external_test_csv")
    test_path = resolve(test_arg) if test_arg else None

    df = pd.read_csv(train_path)
    if "split" not in df.columns:
        raise SystemExit(f"{train_path} must have a 'split' column (0=train, 1=val).")
    drop = [c for c in M.DROP_COLS if c in df.columns]
    train = df[df["split"] == 0].drop(columns=drop + ["split"])
    val = df[df["split"] == 1].drop(columns=drop + ["split"])
    test = None
    if test_path:
        tdf = pd.read_csv(test_path)
        test = tdf.drop(columns=[c for c in M.DROP_COLS + ["split"] if c in tdf.columns])

    imaging_cols = M.detect_imaging_cols(train.columns)
    feats = M.build_feature_list(imaging_cols, mcfg)
    clin_added = [c for c in feats if c not in imaging_cols]
    models = ([s.strip() for s in args.models.split(",") if s.strip()] if args.models
              else list(mcfg.get("models") or []) or M.available_models())
    print(f"[model] train={train.shape} val={val.shape} "
          f"test={None if test is None else test.shape} | "
          f"features={len(feats)} ({len(imaging_cols)} imaging + {len(clin_added)} clinical: "
          f"{clin_added or 'none'}) | models={models} | mode={args.mode}")

    table = M.run_all(feats, train, val, test, models, mcfg,
                      mode=args.mode, tabpfn_phe=args.tabpfn_phe)
    sort_key = "val_RMSE" if "val_RMSE" in table.columns else "cv_rmse_mean"
    table = table.sort_values(sort_key)
    out_path = os.path.join(results_dir, "modeling_results.csv")
    table.to_csv(out_path, index=False)

    print(f"\nResults (target={mcfg['target']}, metrics on original scale), sorted by {sort_key}:\n")
    fmt = {c: "{:.4f}".format for c in table.columns if table[c].dtype.kind == "f"}
    print(table.to_string(index=False, formatters=fmt))
    print(f"\n[model] saved -> {out_path}")


if __name__ == "__main__":
    main()

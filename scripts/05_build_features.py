#!/usr/bin/env python
"""Merge the radiomics (step 3) and shape (step 4) outputs into ONE feature set.

Produces results/feature_table.csv: one row per case with the 14 imaging
variables (11 ``original_shape_*`` averaged over left/right, plus
``convexity_ratio_area``, ``convexity_ratio_vol``, ``local_thickness_mean``).

Usage:
    python scripts/05_build_features.py
"""

from __future__ import annotations

import os

from _common import load_config, resolve


def main():
    cfg = load_config()
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=None, help="dir with step 3/4 CSVs")
    args = p.parse_args()

    from kidney_radiomics.modeling import IMAGING_FEATURES, build_feature_table

    results_dir = resolve(args.results or cfg["output"]["dir"])
    table = build_feature_table(results_dir)
    out_path = os.path.join(results_dir, "feature_table.csv")
    table.to_csv(out_path, index=False)

    present = [f for f in IMAGING_FEATURES if f in table.columns]
    print(f"[features] {table.shape[0]} cases x {len(present)} imaging variables "
          f"(L/R averaged, no BSA) -> {out_path}")
    for f in present:
        print(f"    {f}")


if __name__ == "__main__":
    main()

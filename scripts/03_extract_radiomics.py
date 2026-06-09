#!/usr/bin/env python
"""Extract PyRadiomics SHAPE features (left / right / total kidney) to CSV.

Only morphological shape features are extracted and used in this study; no
first-order, texture, LoG, or Wavelet features are computed.

Usage:
    python scripts/03_extract_radiomics.py
    python scripts/03_extract_radiomics.py --masks results/pred_masks
"""

from __future__ import annotations

import argparse
import os

from _common import load_config, resolve


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", default=None)
    p.add_argument("--masks", default=None)
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()

    from kidney_radiomics.radiomics_features import extract_dataset

    images_dir = resolve(args.images or cfg["data"]["images_dir"])
    masks_dir = resolve(args.masks or cfg["data"]["masks_dir"])
    params_file = resolve(cfg["radiomics"]["params"])
    out_dir = resolve(args.out or cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)

    tables = extract_dataset(
        images_dir, masks_dir, params_file,
        mask_suffix=cfg["data"]["mask_suffix"], extension=cfg["data"]["extension"],
    )

    for side, df in tables.items():
        out_path = os.path.join(out_dir, f"radiomics_shape_{side}.csv")
        df.to_csv(out_path, index=False)
        print(f"[radiomics] {side}: {df.shape[0]} cases x {df.shape[1]-1} shape features -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract convex-hull / local-thickness intermediates to CSV.

Computes, in mm units and averaged over the two kidneys: convex-hull volume,
convex-hull area, and mean local thickness. Convexity and solidity are formed
later in step 05 from these and the radiomics shape features.

Usage:
    python scripts/04_extract_shape_features.py
    python scripts/04_extract_shape_features.py --masks results/pred_masks
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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from kidney_radiomics.shape_features import extract_dataset

    images_dir = resolve(args.images or cfg["data"]["images_dir"])
    masks_dir = resolve(args.masks or cfg["data"]["masks_dir"])
    out_dir = resolve(args.out or cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)

    df = extract_dataset(
        images_dir, masks_dir,
        mask_suffix=cfg["data"]["mask_suffix"], extension=cfg["data"]["extension"],
    )
    out_path = os.path.join(out_dir, "shape_features.csv")
    df.to_csv(out_path, index=False)
    print(f"[shape] {df.shape[0]} cases x {df.shape[1]-1} features -> {out_path}")


if __name__ == "__main__":
    main()

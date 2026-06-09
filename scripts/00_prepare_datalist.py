#!/usr/bin/env python
"""Build a MONAI Decathlon-style datalist (train/val/test) from your data dirs.

Scans the configured images/masks directories, pairs each image with its mask,
and writes dataset.json for segmentation training. No data is generated.

Usage:
    python scripts/00_prepare_datalist.py --val-frac 0.15 --test-frac 0.15
"""

from __future__ import annotations

import argparse
import json

from _common import load_config, resolve

from kidney_radiomics.io_utils import list_cases


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    args = p.parse_args()

    d = cfg["data"]
    images_dir = resolve(d["images_dir"])
    masks_dir = resolve(d["masks_dir"])
    cases = list_cases(images_dir, masks_dir, d["mask_suffix"], d["extension"])
    if not cases:
        raise FileNotFoundError(f"No image/mask pairs in {images_dir} / {masks_dir}.")

    items = [{"image": ip, "label": mp} for _cid, ip, mp in cases]
    n = len(items)
    n_val = max(1, int(round(n * args.val_frac)))
    n_test = max(1, int(round(n * args.test_frac)))
    datalist = {
        "labels": {"0": "background", "1": "kidney"},
        "tensorImageSize": "3D",
        "training": items[: n - n_val - n_test],
        "validation": items[n - n_val - n_test: n - n_test],
        "test": items[n - n_test:],
    }
    out = resolve(d["datalist_json"])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=2)
    print(f"{n} cases -> {out}  (train {len(datalist['training'])}, "
          f"val {n_val}, test {n_test})")


if __name__ == "__main__":
    main()

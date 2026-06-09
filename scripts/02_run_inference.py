#!/usr/bin/env python
"""Run segmentation inference, save predicted masks, and report Dice if GT exists.

Usage:
    python scripts/02_run_inference.py --device cpu
    python scripts/02_run_inference.py --images /path/to/images --out results/pred_masks
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from _common import load_config, resolve


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred > 0, gt > 0
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--images", default=None, help="image dir (default: config)")
    p.add_argument("--masks", default=None, help="ground-truth mask dir for Dice (optional)")
    p.add_argument("--out", default=None, help="output dir for predicted masks")
    args = p.parse_args()

    import torch

    from kidney_radiomics.io_utils import (
        binarize, case_id_from_path, read_volume, write_mask,
    )
    from kidney_radiomics.segmentation import build_model, predict_volume

    seg_cfg = dict(cfg["segmentation"])
    seg_cfg["checkpoint"] = resolve(seg_cfg["checkpoint"])
    images_dir = resolve(args.images or cfg["data"]["images_dir"])
    masks_dir = resolve(args.masks) if args.masks else resolve(cfg["data"]["masks_dir"])
    out_dir = resolve(args.out or os.path.join(cfg["output"]["dir"], "pred_masks"))
    extension = cfg["data"]["extension"]
    mask_suffix = cfg["data"]["mask_suffix"]
    os.makedirs(out_dir, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = build_model(seg_cfg["in_channels"], seg_cfg["out_channels"],
                        seg_cfg["feature_size"], device)
    if os.path.exists(seg_cfg["checkpoint"]):
        model.load_state_dict(torch.load(seg_cfg["checkpoint"], map_location=device))
        print(f"[infer] loaded checkpoint {seg_cfg['checkpoint']}")
    else:
        print(f"[infer] WARNING: no checkpoint at {seg_cfg['checkpoint']}; predictions will be untrained")

    dices = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(extension):
            continue
        cid = case_id_from_path(fname)
        image_path = os.path.join(images_dir, fname)
        pred = predict_volume(model, image_path, seg_cfg, device=device)

        ref = read_volume(image_path).image
        out_path = os.path.join(out_dir, f"{cid}{mask_suffix}{extension}")
        write_mask(pred, ref, out_path)

        gt_path = os.path.join(masks_dir, f"{cid}{mask_suffix}{extension}")
        if os.path.exists(gt_path):
            gt = binarize(read_volume(gt_path).array)
            d = dice_score(pred, gt)
            dices.append(d)
            print(f"[infer] {cid}: Dice={d:.4f} -> {out_path}")
        else:
            print(f"[infer] {cid}: saved -> {out_path}")

    if dices:
        print(f"[infer] mean Dice over {len(dices)} cases: {np.mean(dices):.4f}")


if __name__ == "__main__":
    main()

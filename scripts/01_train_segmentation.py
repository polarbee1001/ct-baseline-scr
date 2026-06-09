#!/usr/bin/env python
"""Train the SwinUNETR kidney segmentation model.

Defaults come from config.yaml (minimal; study values are in its comments). Use
--max-iters / --device for a quick smoke run.

Usage:
    python scripts/01_train_segmentation.py --max-iters 50 --device cpu
"""

from __future__ import annotations

import argparse

from _common import load_config, resolve


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-iters", type=int, default=None, help="override training iterations")
    p.add_argument("--eval-interval", type=int, default=None)
    args = p.parse_args()

    # Import after path setup so MONAI/torch are only required for this stage.
    from kidney_radiomics.segmentation import train

    seg_cfg = dict(cfg["segmentation"])
    # Resolve weight paths relative to the repo root.
    seg_cfg["pretrained_swinvit"] = resolve(seg_cfg["pretrained_swinvit"])
    seg_cfg["checkpoint"] = resolve(seg_cfg["checkpoint"])

    datalist = resolve(cfg["data"]["datalist_json"])
    train(datalist, seg_cfg, device=args.device,
          max_iters=args.max_iters, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()

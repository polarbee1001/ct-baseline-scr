"""Kidney segmentation with a MONAI SwinUNETR model.

Model factory, train/validation transforms, a compact training loop, and
sliding-window inference with largest-connected-component post-processing.
Hyper-parameters are read from the config block passed in.

The self-supervised Swin backbone (``model_swinvit.pt``) is public (MONAI model
zoo); fine-tuned weights are not distributed.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from monai.data import CacheDataset, ThreadDataLoader, decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
)
from skimage.measure import label


def build_model(
    in_channels: int = 1,
    out_channels: int = 2,
    feature_size: int = 48,
    device: str = "cuda",
) -> SwinUNETR:
    """Instantiate a SwinUNETR on the given device.

    MONAI >=1.5 infers the spatial size at forward time (no ``img_size`` arg).
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    )
    return model.to(device)


def load_pretrained_backbone(model: SwinUNETR, weights_path: str) -> None:
    """Load the public self-supervised Swin ViT backbone, if available."""
    if weights_path and os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location="cpu")
        model.load_from(weights=weights)
        print(f"[seg] loaded pretrained backbone from {weights_path}")
    else:
        print(f"[seg] backbone weights not found ({weights_path}); training from scratch")


def train_transforms(hu_window=(-175, 250), roi_size=(96, 96, 96)):
    a_min, a_max = hu_window
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=tuple(roi_size),
            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        EnsureTyped(keys=["image", "label"]),
    ])


def val_transforms(hu_window=(-175, 250)):
    a_min, a_max = hu_window
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"]),
    ])


def infer_transforms(hu_window=(-175, 250)):
    """Image-only transforms for inference on unlabelled volumes.

    No spatial reorientation/cropping is applied so the predicted array stays on
    the input voxel grid and can be saved with the source image geometry. This
    assumes inputs are already isotropic and RAS-oriented (the upstream
    preprocessing convention); otherwise add Orientationd + Invertd.
    """
    a_min, a_max = hu_window
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])


def train(
    datalist_json: str,
    cfg: dict,
    device: str = "cuda",
    max_iters: int | None = None,
    eval_interval: int | None = None,
):
    """Train SwinUNETR and save the best (highest val Dice) checkpoint.

    ``cfg`` is the ``segmentation`` block of config.yaml. ``max_iters`` /
    ``eval_interval`` override the config (handy for a short smoke run).
    """
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    roi = tuple(cfg["roi_size"])
    hu = tuple(cfg["hu_window"])
    max_iters = max_iters or cfg["max_iters"]
    eval_interval = eval_interval or cfg["eval_interval"]

    train_files = load_decathlon_datalist(datalist_json, True, "training")
    val_files = load_decathlon_datalist(datalist_json, True, "validation")

    # cache_rate trades RAM for speed (study: 1.0; default 0.0 = low memory).
    cache_rate = cfg.get("cache_rate", 0.0)
    train_ds = CacheDataset(data=train_files, transform=train_transforms(hu, roi),
                            cache_rate=cache_rate, num_workers=2)
    val_ds = CacheDataset(data=val_files, transform=val_transforms(hu),
                          cache_rate=cache_rate, num_workers=2)
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    model = build_model(cfg["in_channels"], cfg["out_channels"], cfg["feature_size"], device)
    load_pretrained_backbone(model, cfg.get("pretrained_swinvit", ""))

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_label = AsDiscrete(to_onehot=cfg["out_channels"])
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg["out_channels"])

    os.makedirs(os.path.dirname(os.path.abspath(cfg["checkpoint"])), exist_ok=True)

    best_dice = -1.0
    step = 0
    model.train()
    while step < max_iters:
        for batch in train_loader:
            step += 1
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % eval_interval == 0 or step >= max_iters:
                dice = _validate(model, val_loader, roi, cfg["sw_batch_size"],
                                 cfg["infer_overlap"], device, dice_metric,
                                 post_label, post_pred)
                print(f"[seg] iter {step}/{max_iters}  loss={loss.item():.4f}  val_dice={dice:.4f}")
                if dice > best_dice:
                    best_dice = dice
                    torch.save(model.state_dict(), cfg["checkpoint"])
                    print(f"[seg] saved best checkpoint (dice={best_dice:.4f}) -> {cfg['checkpoint']}")
                model.train()
            if step >= max_iters:
                break

    print(f"[seg] training done. best val dice = {best_dice:.4f}")
    return best_dice


def _validate(model, loader, roi, sw_batch, overlap, device, metric, post_label, post_pred):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = sliding_window_inference(x, roi, sw_batch, model, overlap=overlap)
            preds = [post_pred(p) for p in decollate_batch(logits)]
            labels = [post_label(l) for l in decollate_batch(y)]
            metric(y_pred=preds, y=labels)
    return float(metric.aggregate().item())


def predict_volume(model, image_path: str, cfg: dict, device: str = "cuda") -> np.ndarray:
    """Run sliding-window inference on one NIfTI image; return a (z, y, x) mask.

    The prediction is taken back to the input voxel grid and the largest
    connected component is kept (single-organ post-processing per side is left
    to the feature stage).
    """
    roi = tuple(cfg["roi_size"])
    transform = infer_transforms(tuple(cfg["hu_window"]))
    data = transform({"image": image_path})
    x = data["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = sliding_window_inference(x, roi, cfg["sw_batch_size"], model,
                                              overlap=cfg["infer_overlap"])
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    # MONAI keeps spatial axes as (x, y, z); io_utils / write_mask use SimpleITK's
    # (z, y, x). Transpose so the saved mask aligns with the source geometry.
    pred = np.transpose(pred, (2, 1, 0))
    return keep_largest_components(pred, n=2)


def keep_largest_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    """Keep the ``n`` largest connected components (expected: two kidneys)."""
    components = label(mask > 0, connectivity=3)
    labels, counts = np.unique(components, return_counts=True)
    fg = labels != 0
    labels, counts = labels[fg], counts[fg]
    if labels.size == 0:
        return mask
    keep = labels[np.argsort(counts)[::-1][:n]]
    return np.isin(components, keep).astype(np.uint8)

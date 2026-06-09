"""PyRadiomics SHAPE feature extraction for the left and right kidney.

Only morphological shape features are extracted (see params/radiomics_shape.yaml);
no first-order, texture, LoG or Wavelet features. Each case's CT image is paired
with the left- and right-kidney masks. No HU windowing is applied; PyRadiomics
reads voxel spacing from the image header, so all physical features are in mm.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

from .io_utils import list_cases, read_volume, separate_kidneys

# PyRadiomics is very chatty; keep only warnings/errors.
logging.getLogger("radiomics").setLevel(logging.ERROR)


def _mask_image(mask_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Wrap a (z, y, x) array as a SimpleITK mask sharing ``reference`` geometry."""
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.CopyInformation(reference)
    return img


def _extract_one(extractor, image: sitk.Image, mask_zyx: np.ndarray):
    """Run the extractor for a single mask, returning an ordered dict of features."""
    mask_img = _mask_image(mask_zyx, image)
    result = extractor.execute(image, mask_img)
    # Drop diagnostics_* keys (provenance, not features).
    return {k: v for k, v in result.items() if not k.startswith("diagnostics_")}


def extract_dataset(
    images_dir: str,
    masks_dir: str,
    params_file: str,
    mask_suffix: str = "_mask",
    extension: str = ".nii.gz",
):
    """Extract features for every case.

    Returns a dict with keys ``"left"`` and ``"right"``; each row is one case.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    cases = list_cases(images_dir, masks_dir, mask_suffix, extension)
    if not cases:
        raise FileNotFoundError(f"No image/mask pairs found in {images_dir}.")

    rows = {"left": [], "right": []}
    for cid, image_path, mask_path in tqdm(cases, desc="radiomics"):
        vol = read_volume(image_path)
        mask_left, mask_right = separate_kidneys(read_volume(mask_path).array)
        for side, m in (("left", mask_left), ("right", mask_right)):
            feats = _extract_one(extractor, vol.image, m)
            rows[side].append({"case_id": cid, **{k: float(v) for k, v in feats.items()}})

    return {side: pd.DataFrame(r) for side, r in rows.items()}

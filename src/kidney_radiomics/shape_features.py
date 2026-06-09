"""Convex-hull and local-thickness intermediates.

For each kidney (left and right) this stage produces, in physical units (mm):

  * ``hull_volume``           -- volume of the 3D convex hull (mm^3)
  * ``hull_area``             -- surface area of the 3D convex hull (mm^2)
  * ``local_thickness_mean``  -- mean local thickness (mm)

Hull volume/area come from :class:`scipy.spatial.ConvexHull`; values are
averaged over the two kidneys. These are intermediates: the build step
(:mod:`kidney_radiomics.modeling`) combines them with the PyRadiomics shape
features to form ``convexity`` and ``solidity``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm

from .io_utils import list_cases, read_volume, separate_kidneys

try:
    import localthickness as lt
    _HAS_LT = True
except Exception:  # pragma: no cover - optional dependency (needs Python >= 3.9)
    _HAS_LT = False


def hull_volume_area(mask_zyx: np.ndarray, spacing_zyx) -> tuple[float, float]:
    """Volume (mm^3) and surface area (mm^2) of the mask's 3D convex hull."""
    coords = np.argwhere(mask_zyx > 0).astype(float) * np.asarray(spacing_zyx)
    hull = ConvexHull(coords)
    return float(hull.volume), float(hull.area)


def mean_local_thickness(mask_zyx: np.ndarray, spacing_zyx) -> float:
    """Mean local thickness (mm): average inscribed-sphere radius in the kidney.

    The mask is cropped to its bounding box first (thickness is distance-based,
    so surrounding empty space does not change it) for speed. Returns NaN if the
    ``localthickness`` package is unavailable.
    """
    if not _HAS_LT:
        return float("nan")
    idx = np.argwhere(mask_zyx > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    crop = mask_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    thickness = lt.local_thickness(crop.astype(np.float32), scale=1)
    nonzero = thickness[thickness > 0]
    if nonzero.size == 0:
        return float("nan")
    return float(np.mean(nonzero) * np.mean(spacing_zyx))


def _side_features(mask_zyx: np.ndarray, spacing_zyx) -> dict:
    vol, area = hull_volume_area(mask_zyx, spacing_zyx)
    return {
        "hull_volume": vol,
        "hull_area": area,
        "local_thickness_mean": mean_local_thickness(mask_zyx, spacing_zyx),
    }


def extract_dataset(
    images_dir: str,
    masks_dir: str,
    mask_suffix: str = "_mask",
    extension: str = ".nii.gz",
) -> pd.DataFrame:
    """Compute hull/thickness intermediates per case, averaged over both kidneys.

    Columns: ``case_id, hull_volume, hull_area, local_thickness_mean``
    (each the mean of the left and right kidney). The image directory is only
    used to enumerate cases; spacing is read from the mask header.
    """
    cases = list_cases(images_dir, masks_dir, mask_suffix, extension)
    if not cases:
        raise FileNotFoundError(f"No image/mask pairs found in {images_dir}.")
    if not _HAS_LT:
        print("[warn] 'localthickness' not installed; thickness column will be NaN.")

    keys = ("hull_volume", "hull_area", "local_thickness_mean")
    rows = []
    for cid, _image_path, mask_path in tqdm(cases, desc="shape"):
        mask = read_volume(mask_path)
        left, right = separate_kidneys(mask.array)
        fl = _side_features(left, mask.spacing_zyx)
        fr = _side_features(right, mask.spacing_zyx)
        rows.append({"case_id": cid, **{k: (fl[k] + fr[k]) / 2.0 for k in keys}})
    return pd.DataFrame(rows)

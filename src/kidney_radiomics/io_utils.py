"""I/O helpers and the shared left/right kidney separation routine.

This module is the single source of truth for:
  * reading / writing NIfTI volumes with SimpleITK,
  * recovering voxel spacing (mm),
  * splitting a bilateral kidney mask into left and right components.

Both the radiomics and shape-feature stages import ``separate_kidneys`` so the
left/right convention is guaranteed to be identical across feature sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from skimage.measure import label

# SimpleITK arrays are returned in (z, y, x) order; spacing from GetSpacing()
# is (x, y, z). Helpers below keep this conversion explicit.


@dataclass
class Volume:
    """A loaded NIfTI volume: ``array`` (z, y, x), ``spacing_zyx`` in mm, and the
    source SimpleITK ``image`` (carries geometry)."""

    array: np.ndarray
    spacing_zyx: tuple
    image: sitk.Image


def read_volume(path: str) -> Volume:
    """Read a NIfTI file into a :class:`Volume`."""
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    sx, sy, sz = image.GetSpacing()  # (x, y, z)
    return Volume(array=array, spacing_zyx=(sz, sy, sx), image=image)


def write_mask(array_zyx: np.ndarray, reference: sitk.Image, path: str) -> None:
    """Write an integer label array using ``reference`` geometry."""
    out = sitk.GetImageFromArray(array_zyx.astype(np.uint8))
    out.CopyInformation(reference)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sitk.WriteImage(out, path)


def case_id_from_path(path: str) -> str:
    """Derive a case identifier from a NIfTI filename (strips .nii/.nii.gz)."""
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def binarize(mask: np.ndarray) -> np.ndarray:
    """Collapse any positive label to 1 (uint8)."""
    return (mask > 0).astype(np.uint8)


def separate_kidneys(mask: np.ndarray):
    """Split a bilateral kidney mask (z, y, x) into (left, right) binary masks.

    The two largest 26-connected components are the kidneys; the one with the
    larger minimum x index is the patient's left. Raises ValueError if fewer
    than two foreground components are present.
    """
    binary = binarize(mask)
    components = label(binary, connectivity=3)
    labels, counts = np.unique(components, return_counts=True)

    # Drop background (label 0) before ranking by size.
    fg = labels != 0
    labels, counts = labels[fg], counts[fg]
    if labels.size < 2:
        raise ValueError(
            f"Expected two kidney components, found {labels.size}. "
            "Check the mask or your left/right assumptions."
        )

    # Two largest components are the kidneys.
    order = np.argsort(counts)[::-1]
    lab_a, lab_b = labels[order[0]], labels[order[1]]

    min_x_a = np.where(components == lab_a)[2].min()
    min_x_b = np.where(components == lab_b)[2].min()
    if min_x_a > min_x_b:
        left_lab, right_lab = lab_a, lab_b
    else:
        left_lab, right_lab = lab_b, lab_a

    mask_left = (components == left_lab).astype(np.uint8)
    mask_right = (components == right_lab).astype(np.uint8)
    return mask_left, mask_right


def list_cases(images_dir: str, masks_dir: str, mask_suffix: str, extension: str):
    """Yield (case_id, image_path, mask_path) for every image with a mask.

    Cases whose mask file is missing are skipped (a warning is the caller's
    responsibility). Ordering is sorted by case id for determinism.
    """
    cases = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(extension):
            continue
        cid = case_id_from_path(fname)
        image_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, f"{cid}{mask_suffix}{extension}")
        if os.path.exists(mask_path):
            cases.append((cid, image_path, mask_path))
    return cases

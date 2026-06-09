"""Kidney CT radiomics pipeline (public release).

Modules
-------
io_utils          : NIfTI I/O, voxel spacing, left/right kidney separation.
segmentation      : MONAI SwinUNETR training and sliding-window inference.
radiomics_features: PyRadiomics shape feature extraction (left / right / total).
shape_features    : Convex-hull volume/area + mean local thickness.
modeling          : Imaging-feature assembly + regression pipeline
                    (log target, RepeatedStratifiedKFold, model panel, HPO).

No patient data is bundled with this package. Provide your own NIfTI images,
masks, and clinical table (see README for the expected layout).
"""

__version__ = "1.0.0"

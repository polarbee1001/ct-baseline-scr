# Kidney CT Radiomics Pipeline

Reference implementation: deep-learning kidney segmentation, morphological
**shape** feature extraction, and regression modeling of baseline renal function.

**No patient data is included** — provide your own NIfTI images, masks, and a
clinical table (formats below).

## Pipeline (scripts 00–06)

1. **Segmentation** — MONAI `SwinUNETR`: training + sliding-window inference.
2. **Radiomics** — PyRadiomics shape features (11; no first-order/texture/filters).
3. **Shape features** — convex hull (convexity, solidity) + mean local thickness.
4. **Combine** — merge steps 2–3, averaging each variable over the left & right kidney.
5. **Modeling** — regression with a model panel (+ optional Optuna HPO / TabPFN PHE).

## Imaging features (14)

11 PyRadiomics shape features + `convexity`, `solidity`, `local_thickness_mean`,
each averaged over both kidneys (the bilateral mask is split by connected
components):

`original_shape_{MeshVolume, SurfaceArea, SurfaceVolumeRatio, Sphericity,
Maximum3DDiameter, Maximum2DDiameterColumn, Maximum2DDiameterSlice,
MinorAxisLength, LeastAxisLength, Elongation, Flatness}`

- **convexity** = convex-hull surface area / ROI surface area
- **solidity** = ROI volume / convex-hull volume
- **local_thickness_mean** = mean local thickness within the kidney (mm)

## Environment

One Python 3.9 conda environment runs everything:

```bash
conda env create -f environment.yml
conda activate kidney-radiomics
```

`pyradiomics` is built from source (`environment.yml` adds a C/C++ compiler);
all other packages come from `requirements.txt`.

## Input data

```
data/images/<case_id>.nii.gz       # isotropic CT
data/masks/<case_id>_mask.nii.gz   # bilateral kidney mask (any positive value = kidney)
```

For modeling, supply a CSV that contains the imaging features + clinical
variables + target, with a 0/1 `split` column (0 = train, 1 = val); optionally a
separate external-test CSV of the same schema.

## Usage

```bash
python scripts/00_prepare_datalist.py          # segmentation datalist
python scripts/01_train_segmentation.py
python scripts/02_run_inference.py --out results/pred_masks
python scripts/03_extract_radiomics.py         # radiomics shape (left/right)
python scripts/04_extract_shape_features.py    # convex hull + local thickness
python scripts/05_build_features.py            # merge + L/R average -> feature_table.csv
python scripts/06_run_modeling.py --train data/train_val.csv --test data/external_test.csv
python scripts/06_run_modeling.py --mode hpo   # Optuna HPO  (--tabpfn-phe adds PHE)
```

Outputs in `results/`: `radiomics_shape_{left,right}.csv`, `shape_features.csv`,
`feature_table.csv`, `modeling_results.csv`.

## Modeling

- Target modelled in log space; metrics (RMSE / MAE / MBE / MAPE) on the
  back-transformed (`exp`) original scale.
- Feature set = 14 imaging + the clinical columns in `modeling.clinical_columns`
  (empty → imaging only).
- Models: LinearRegression, LASSO, ElasticNet, SVR, RandomForest, XGBoost,
  LightGBM, and TabPFN (default / HPO / PHE). HPO uses Optuna over
  `RepeatedStratifiedKFold(5, 3)`.
- TabPFN needs a one-time license: set `TABPFN_TOKEN`
  (https://ux.priorlabs.ai) or point `modeling.tabpfn_model_path` to a local
  checkpoint; the other models run without it.

All settings live in **`config.yaml`**.

## Model weights

The self-supervised Swin ViT backbone (`model_swinvit.pt`, MONAI model zoo) goes
in `weights/`. Fine-tuned (patient-trained) checkpoints are not distributed.

## Layout

```
public/
├── config.yaml                 # all settings
├── environment.yml             # single conda env (py3.9)
├── params/radiomics_shape.yaml # PyRadiomics shape params (11 features)
├── src/kidney_radiomics/       # io_utils, segmentation, radiomics_features,
│                               #   shape_features, modeling
└── scripts/                    # CLI entry points 00–06
```

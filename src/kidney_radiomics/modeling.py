"""Feature assembly and regression modeling.

1. **Feature assembly.** ``build_feature_table`` merges the step-3 radiomics
   output (``original_shape_*``, averaged over left/right) with the step-4
   hull/thickness intermediates into the 14 imaging variables:

       original_shape_* (11)   -- PyRadiomics shape features
       convexity               -- convex-hull area / ROI surface area
       solidity                -- ROI volume / convex-hull volume
       local_thickness_mean    -- mean local thickness (mm)

2. **Modeling.** The target is modelled in log space, with metrics reported on
   the back-transformed (``exp``) original scale. Hyper-parameters are tuned with
   Optuna (minimising RepeatedStratifiedKFold CV RMSE); models are
   LinearRegression, LASSO, ElasticNet, SVR, RandomForest, XGBoost, LightGBM and
   optionally TabPFN; metrics are RMSE / MAE / MBE / MAPE. One feature set is
   used: the imaging variables plus any clinical columns from ``config.yaml``
   (``modeling.clinical_columns``; empty -> imaging only).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---- optional dependencies -------------------------------------------------
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False
try:
    from tabpfn import TabPFNRegressor  # tabpfn==7.1.1 -> model v2.5
    _HAS_TABPFN = True
except Exception:
    _HAS_TABPFN = False
try:  # TabPFN HPO / post-hoc ensembling (optional extra package)
    from tabpfn_extensions.hpo import TunedTabPFNRegressor
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    _HAS_TABPFN_EXT = True
except Exception:
    _HAS_TABPFN_EXT = False


def _patch_tabpfn_license():
    """tabpfn 7.1.1 builds the HuggingFace license URL with a wrong repo prefix,
    which 404s and blocks model download. Normalise it once (no-op if absent)."""
    try:
        from tabpfn import browser_auth as ba
    except Exception:
        return
    import json
    import urllib.request

    def _name(repo_id):
        repo = repo_id if repo_id.startswith("Prior-Labs/") else f"Prior-Labs/{repo_id}"
        try:
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=10) as r:
                return json.loads(r.read()).get("cardData", {}).get("license_name")
        except Exception:
            return None
    ba._get_license_name = _name


_patch_tabpfn_license()

SEED = 42

# Search space for TabPFN HPO (TunedTabPFNRegressor).
TABPFN_HPO_SPACE = {
    "model_type": ["single"],
    "n_estimators": [4],
    "average_before_softmax": [True, False],
    "softmax_temperature": [0.9, 1.0, 1.05],
    "inference_config/FINGERPRINT_FEATURE": [True, False],
    "inference_config/POLYNOMIAL_FEATURES": ["no"],
    "inference_config/OUTLIER_REMOVAL_STD": [None, 12.0],
    "inference_config/MIN_UNIQUE_FOR_NUMERICAL_FEATURES": [10, 30],
    "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": [(None,), (None, "safepower")],
}

# ===========================================================================
# 1. Feature assembly
# ===========================================================================
# The 11 retained PyRadiomics shape features (params/radiomics_shape.yaml).
RADIOMICS_SHAPE = [
    "original_shape_MeshVolume",
    "original_shape_SurfaceArea",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_Sphericity",
    "original_shape_Maximum3DDiameter",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_MinorAxisLength",
    "original_shape_LeastAxisLength",
    "original_shape_Elongation",
    "original_shape_Flatness",
]
DERIVED = ["convexity", "solidity", "local_thickness_mean"]
IMAGING_FEATURES = RADIOMICS_SHAPE + DERIVED  # 14 imaging variables


def _avg_radiomics(results_dir, prefix="radiomics_shape"):
    """Average the per-side radiomics shape tables into one per-case table."""
    left = pd.read_csv(os.path.join(results_dir, f"{prefix}_left.csv")).set_index("case_id")
    right = pd.read_csv(os.path.join(results_dir, f"{prefix}_right.csv")).set_index("case_id")
    cols = [c for c in RADIOMICS_SHAPE if c in left.columns]
    return (left[cols] + right[cols]) / 2.0


def build_feature_table(results_dir, radiomics_prefix="radiomics_shape",
                        shape_file="shape_features.csv"):
    """Merge steps 3 & 4 into the single imaging feature table.

    Returns one row per case with ``case_id`` + the 14 imaging variables, plus
    the hull intermediates (``hull_volume``/``hull_area``) for traceability.

    Convexity = convex-hull area / ROI surface area; both are <= 1 for a smooth
    object. Solidity = ROI volume / convex-hull volume.
    """
    radiomics = _avg_radiomics(results_dir, radiomics_prefix)
    shape = pd.read_csv(os.path.join(results_dir, shape_file)).set_index("case_id")
    tbl = radiomics.join(shape, how="inner")
    tbl["convexity"] = tbl["hull_area"] / tbl["original_shape_SurfaceArea"]
    tbl["solidity"] = tbl["original_shape_MeshVolume"] / tbl["hull_volume"]
    return tbl.reset_index()


# Columns dropped before modelling: identifiers, the target's untransformed
# source, and shape descriptors redundant with the retained features.
DROP_COLS = [
    "A_NUM", "CT_DATE", "Cr_48h", "Baseline_Cr",
    "original_shape_VoxelVolume", "original_shape_Compactness1",
    "original_shape_Compactness2", "original_shape_SphericalDisproportion",
    "original_shape_Maximum2DDiameterRow", "original_shape_MajorAxisLength",
]


def detect_imaging_cols(columns) -> list:
    """Imaging columns, identified by their naming convention."""
    return [c for c in columns
            if c.startswith("original_shape_")
            or c in ("convexity", "solidity", "local_thickness_mean")]


def build_feature_list(imaging_cols, cfg) -> list:
    """The single modeling feature set: imaging + chosen clinical columns.

    ``cfg['clinical_columns']`` lists the clinical variables to append to the
    detected imaging features (empty -> imaging only). The target is never
    included as a feature.
    """
    clin = [c for c in (cfg.get("clinical_columns") or []) if c != cfg["target"]]
    return list(imaging_cols) + clin


# ===========================================================================
# 2. Modeling
# ===========================================================================
def available_models() -> list:
    names = ["LinearRegression", "LASSO", "ElasticNet", "SVR", "RandomForest"]
    if _HAS_XGB:
        names.append("XGBoost")
    if _HAS_LGB:
        names.append("LightGBM")
    if _HAS_TABPFN:
        names.append("TabPFN")
    return names


def build_model(name, params=None):
    """Instantiate a model (StandardScaler-wrapped for linear/SVR)."""
    params = params or {}
    if name == "LinearRegression":
        est = LinearRegression()
    elif name == "LASSO":
        est = Lasso(max_iter=50_000, tol=1e-5, random_state=SEED, **params)
    elif name == "ElasticNet":
        est = ElasticNet(max_iter=50_000, tol=1e-5, random_state=SEED, **params)
    elif name == "SVR":
        est = SVR(**params)
    elif name == "RandomForest":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
    elif name == "XGBoost":
        return XGBRegressor(n_jobs=-1, random_state=SEED, verbosity=0, **params)
    elif name == "LightGBM":
        return LGBMRegressor(n_jobs=-1, random_state=SEED, verbose=-1, **params)
    elif name == "TabPFN":
        return TabPFNRegressor(random_state=SEED, ignore_pretraining_limits=True)
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("model", est)])


def _suggest(name, trial):
    """Optuna search space for each model."""
    if name == "LASSO":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True)}
    if name == "ElasticNet":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    if name == "SVR":
        return {"C": trial.suggest_float("C", 1e-1, 1e3, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"])}
    if name == "RandomForest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])}
    if name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 8, 1024, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-2, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0)}
    if name == "LightGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 8, 1024, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024, log=True),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 150),
                "cat_smooth": trial.suggest_float("cat_smooth", 1, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True)}
    raise ValueError(f"no search space for {name}")


def metrics(y_true, y_pred, log_target=True) -> dict:
    """RMSE / MAE / MBE / MAPE, on the original scale when ``log_target``."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if log_target:
        yt, yp = np.exp(yt), np.exp(np.clip(yp, -10.0, 10.0))
    return {"RMSE": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "MAE": float(mean_absolute_error(yt, yp)),
            "MBE": float(np.mean(yp - yt)),
            "MAPE": float(mean_absolute_percentage_error(yt, yp))}


def _strata(y):
    return pd.qcut(y, q=5, labels=False, duplicates="drop")


def _cv(cfg):
    return RepeatedStratifiedKFold(n_splits=cfg.get("cv_splits", 5),
                                   n_repeats=cfg.get("cv_repeats", 3),
                                   random_state=SEED)


def cv_rmse(model, X, y, cfg, log_target=True) -> tuple:
    """(mean, sd) of CV RMSE on the original scale (HPO / reporting objective)."""
    rmses = []
    for tr, va in _cv(cfg).split(X, _strata(y)):
        m = clone(model)
        m.fit(X[tr], y[tr])
        rmses.append(metrics(y[va], m.predict(X[va]), log_target)["RMSE"])
    return float(np.mean(rmses)), float(np.std(rmses, ddof=1))


def tune(name, X, y, cfg, log_target=True):
    """Optuna HPO minimising CV RMSE (TPESampler + MedianPruner); best params."""
    if not _HAS_OPTUNA:
        raise RuntimeError("optuna not installed; cannot run HPO.")

    def objective(trial):
        model = build_model(name, _suggest(name, trial))
        return cv_rmse(model, X, y, cfg, log_target)[0]

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(n_startup_trials=10, seed=SEED),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=cfg.get("n_trials", 30), show_progress_bar=False)
    return study.best_params


def build_tabpfn(mode="default", cfg=None):
    """TabPFN regressor for a given mode: default | hpo | phe.

    ``hpo`` and ``phe`` need the ``tabpfn_extensions`` package. A local checkpoint
    (``tabpfn_model_path``) lets the default model skip the download/license step;
    the HPO/PHE wrappers select the model by ``tabpfn_model_version`` instead.
    """
    cfg = cfg or {}
    version = {"model_version": cfg["tabpfn_model_version"]} if cfg.get("tabpfn_model_version") else {}
    if mode == "phe":
        if not _HAS_TABPFN_EXT:
            raise RuntimeError("tabpfn_extensions not installed; cannot run TabPFN PHE.")
        return AutoTabPFNRegressor(
            max_time=cfg.get("phe_time", 120), random_state=SEED,
            ignore_pretraining_limits=True,
            n_ensemble_models=cfg.get("phe_n_models", 20), n_estimators=8,
            presets=cfg.get("phe_presets"), **version)
    if mode == "hpo":
        if not _HAS_TABPFN_EXT:
            raise RuntimeError("tabpfn_extensions not installed; cannot run TabPFN HPO.")
        return TunedTabPFNRegressor(
            n_trials=cfg.get("n_trials", 30), metric="rmse", random_state=SEED,
            search_space=TABPFN_HPO_SPACE, **version)
    path = {"model_path": cfg["tabpfn_model_path"]} if cfg.get("tabpfn_model_path") else {}
    return TabPFNRegressor(random_state=SEED, ignore_pretraining_limits=True, **path)


def _fit_eval(model, label, feats, train, val, test, cfg, compute_cv=True):
    """Fit on train, return a metrics row (val/test metrics, optional CV RMSE).

    CV is skipped for TabPFN: it would refit the (tuned/ensembled) model on every
    fold, which is prohibitively slow and not how TabPFN is evaluated.
    """
    target, log_target = cfg["target"], cfg.get("log_target", True)
    Xtr, ytr = train[feats].values.astype(float), train[target].values
    model.fit(Xtr, ytr)
    row = {"model": label, "n_features": len(feats)}
    if compute_cv:
        row["cv_rmse_mean"], row["cv_rmse_sd"] = cv_rmse(model, Xtr, ytr, cfg, log_target)
    for split_name, frame in (("val", val), ("test", test)):
        if frame is None:
            continue
        m = metrics(frame[target].values,
                    model.predict(frame[feats].values.astype(float)), log_target)
        row.update({f"{split_name}_{k}": v for k, v in m.items()})
    return row


def run_all(feats, train, val, test, model_names, cfg, mode="default", tabpfn_phe=False):
    """Train/evaluate every model on the feature set ``feats``.

    ``train`` is used for HPO + CV; ``val`` and (optional) ``test`` are held-out
    evaluation frames. mode: ``default`` (library defaults) | ``hpo`` (Optuna).
    With ``tabpfn_phe`` an extra TabPFN post-hoc-ensemble row is added.
    Returns a metrics DataFrame, one row per model.
    """
    log_target = cfg.get("log_target", True)
    ytr, Xtr = train[cfg["target"]].values, train[feats].values.astype(float)
    rows = []
    for name in model_names:
        if name == "TabPFN":
            model = build_tabpfn("hpo" if mode == "hpo" else "default", cfg)
            rows.append(_fit_eval(model, name, feats, train, val, test, cfg, compute_cv=False))
            continue
        if mode == "hpo" and name != "LinearRegression":
            model = build_model(name, tune(name, Xtr, ytr, cfg, log_target))
        else:
            model = build_model(name)
        rows.append(_fit_eval(model, name, feats, train, val, test, cfg))
    if tabpfn_phe:
        rows.append(_fit_eval(build_tabpfn("phe", cfg), "TabPFN_PHE",
                              feats, train, val, test, cfg, compute_cv=False))
    return pd.DataFrame(rows)

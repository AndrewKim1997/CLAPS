
# -*- coding: utf-8 -*-
"""
Appendix: Variance-Floor Sensitivity for CLAPS
==============================================

Standalone Google Colab / Python implementation aligned with the revised
real-data benchmark, the matched LACP comparison, and the training-only
prior-precision selection protocol.

Purpose
-------
The heteroscedastic variance floor enters CLAPS twice: it is added to the
learned aleatoric variance head and it lower-bounds the variance used in the
noise-weighted last-layer precision. This experiment therefore retrains the
heteroscedastic regressor for every variance-floor value and evaluates both
interval behavior and numerical stability.

Protocol
--------
1. Use representative real-data datasets with the revised 60/20/20
   train/calibration/test split and training-only standardization.
2. Keep the outer split, fitting/validation split, prior-selection split, and
   neural initialization fixed across variance floors within each
   dataset-seed run.
3. Retrain the heteroscedastic model for every variance floor because the
   floor changes the likelihood and the learned variance head.
4. Select the CLAPS prior precision from a training-only inner split by
   predictive Gaussian negative log-likelihood. Calibration and test labels
   are never used for selection.
5. Compare CLAPS with the matched LACP interval obtained from the same fitted
   model and report effects overall and in the weakest learned-feature support
   quintile.
6. Record lower-tail variance diagnostics, floor contribution, precision
   weights, and the condition number of the selected last-layer precision.

Primary outputs
---------------
results_df
    Per-dataset, per-seed, per-floor LACP and CLAPS results.
selection_trace_df
    Training-only validation NLL for every prior-precision candidate.
floor_performance_summary_df
    Aggregate CLAPS performance and CLAPS-minus-LACP effects by floor.
floor_numerical_summary_df
    Lower-tail variance and last-layer numerical diagnostics by floor.
dataset_floor_summary_df
    Dataset-specific CLAPS sensitivity summary.
lambda_selection_frequency_df
    Prior-precision selection frequency by floor.
floor_bootstrap_df
    Hierarchical paired bootstrap intervals for CLAPS-minus-LACP effects.
reference_floor_bootstrap_df
    Hierarchical paired bootstrap intervals for CLAPS relative to the revised
    main-experiment floor.

Notes
-----
The revised main experiments use variance_floor=1e-4, so that is the default
reference setting here. The grid retains the former appendix range
(1e-6, 1e-5, 1e-4, 1e-3).
"""

from __future__ import annotations

import gc
import math
import random
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# Package setup
# ============================================================


def install_if_missing(import_name: str, package_name: str | None = None) -> None:
    package_name = package_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )


def prepare_packages() -> None:
    install_if_missing("requests")
    install_if_missing("openpyxl")
    install_if_missing("xlrd")
    install_if_missing("sklearn", "scikit-learn")


import requests
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class VarianceFloorConfig:
    # Representative subset retained from the former sensitivity appendix.
    # Replace with all eight revised benchmark datasets when desired:
    # Concrete, Energy, Yacht, Airfoil, WineRed, Naval, Bike, California.
    dataset_names: Tuple[str, ...] = (
        "Concrete",
        "Energy",
        "Bike",
        "California",
    )

    # Final appendix protocol.
    num_seeds: int = 20
    train_fraction: float = 0.60
    cal_fraction: float = 0.20
    test_fraction: float = 0.20
    max_samples_per_dataset: int = 5000

    # Nominal coverage.
    alpha: float = 0.10

    # Shared architecture.
    hidden_dim: int = 64
    feature_dim: int = 64

    # Variance-floor grid. The revised main-experiment value is 1e-4.
    variance_floor_grid: Tuple[float, ...] = (
        1e-6,
        1e-5,
        1e-4,
        1e-3,
    )
    reference_variance_floor: float = 1e-4

    # Neural optimization, matching revised Experiment 4.
    max_epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5

    # Training-only prior-precision selection, matching the main experiment.
    lambda_grid: Tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    )

    # Last-layer numerical regularization is not varied in this experiment.
    jitter: float = 1e-6

    # Independent learned-feature support proxy.
    support_k: int = 20
    support_quintiles: int = 5

    # Hierarchical paired bootstrap.
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 20260801

    # Display and saving.
    verbose: bool = True
    summary_digits: int = 4
    save_csv: bool = True
    output_prefix: str = "variance_floor"
    display_raw_results: bool = False


CFG = VarianceFloorConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("Appendix experiment: variance-floor sensitivity")
print(f"Datasets: {CFG.dataset_names}")
print(f"Variance floors: {CFG.variance_floor_grid}")
print(f"Reference floor: {CFG.reference_variance_floor:g}")
print(f"Prior-precision grid: {CFG.lambda_grid}")


# ============================================================
# Reproducibility and generic utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def deterministic_split(
    n: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("At least two observations are required.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_validation = max(1, int(round(n * validation_fraction)))
    n_validation = min(n_validation, n - 1)
    validation_idx = indices[:n_validation]
    fit_idx = indices[n_validation:]
    return fit_idx, validation_idx


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]

    n = len(scores)
    if n == 0:
        return np.inf

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf

    return float(np.partition(scores, k - 1)[k - 1])


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    width = upper - lower
    score = width.copy()
    below = y < lower
    above = y > upper
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return score


def unstandardize_y(
    y_standardized: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    return y_standardized * y_scale + y_mean


def standardize_with_train(
    x_train: np.ndarray,
    x_cal: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_cal: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_scale = np.std(x_train, axis=0, keepdims=True)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)

    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train) + 1e-8)

    return {
        "x_train": ((x_train - x_mean) / x_scale).astype(np.float32),
        "x_cal": ((x_cal - x_mean) / x_scale).astype(np.float32),
        "x_test": ((x_test - x_mean) / x_scale).astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_cal": y_cal.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "y_train_std": ((y_train - y_mean) / y_scale).astype(np.float32),
        "y_cal_std": ((y_cal - y_mean) / y_scale).astype(np.float32),
        "y_test_std": ((y_test - y_mean) / y_scale).astype(np.float32),
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def safe_sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def format_mean_se(
    mean_value: float,
    sem_value: float,
    digits: int,
    multiplier: float = 1.0,
    suffix: str = "",
) -> str:
    mean_scaled = multiplier * mean_value
    sem_scaled = multiplier * sem_value
    return f"{mean_scaled:.{digits}f} ± {sem_scaled:.{digits}f}{suffix}"


# ============================================================
# Dataset loading
# ============================================================


def clean_xy(
    x_df: pd.DataFrame,
    y_series: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    x_df = pd.DataFrame(x_df).copy()
    y_series = pd.Series(y_series).copy()

    x_df = pd.get_dummies(x_df, drop_first=False)
    for column in x_df.columns:
        x_df[column] = pd.to_numeric(x_df[column], errors="coerce")
    y_series = pd.to_numeric(y_series, errors="coerce")

    combined = x_df.copy()
    combined["_target"] = y_series.to_numpy()
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    y = combined["_target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x = combined.drop(columns=["_target"]).to_numpy(dtype=np.float32)
    return x, y


def load_concrete() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "concrete/compressive/Concrete_Data.xls"
    )
    frame = pd.read_excel(url, engine="xlrd")
    return clean_xy(frame.iloc[:, :-1], frame.iloc[:, -1])


def load_energy_efficiency() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00242/ENB2012_data.xlsx"
    )
    frame = pd.read_excel(url, engine="openpyxl")
    return clean_xy(frame.iloc[:, :8], frame.iloc[:, 8])


def load_yacht_hydrodynamics() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00243/yacht_hydrodynamics.data"
    )
    frame = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    return clean_xy(frame.iloc[:, :-1], frame.iloc[:, -1])


def load_airfoil_self_noise() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00291/airfoil_self_noise.dat"
    )
    frame = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    return clean_xy(frame.iloc[:, :-1], frame.iloc[:, -1])


def load_wine_quality_red() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    frame = pd.read_csv(url, sep=";")
    return clean_xy(frame.drop(columns=["quality"]), frame["quality"])


def load_naval_propulsion() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/static/public/316/"
        "condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(BytesIO(response.content))
    data_filename = next(
        (name for name in archive.namelist() if name.endswith("data.txt")),
        None,
    )
    if data_filename is None:
        raise FileNotFoundError("data.txt was not found in the Naval archive.")

    frame = pd.read_csv(
        archive.open(data_filename),
        sep=r"\s+",
        header=None,
        engine="python",
    ).dropna(axis=1, how="all")
    return clean_xy(frame.iloc[:, :16], frame.iloc[:, 17])


def load_bike_sharing() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00275/Bike-Sharing-Dataset.zip"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(BytesIO(response.content))
    frame = pd.read_csv(archive.open("hour.csv"))

    target = frame["cnt"]
    drop_columns = [
        column
        for column in [
            "cnt",
            "casual",
            "registered",
            "instant",
            "dteday",
        ]
        if column in frame.columns
    ]
    return clean_xy(frame.drop(columns=drop_columns), target)


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing(as_frame=True)
    return clean_xy(data.data, data.target)


def load_all_datasets(
    cfg: VarianceFloorConfig,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    loaders = {
        "Concrete": load_concrete,
        "Energy": load_energy_efficiency,
        "Yacht": load_yacht_hydrodynamics,
        "Airfoil": load_airfoil_self_noise,
        "WineRed": load_wine_quality_red,
        "Naval": load_naval_propulsion,
        "Bike": load_bike_sharing,
        "California": load_california_housing,
    }

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for dataset_name in cfg.dataset_names:
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        print(f"Loading dataset: {dataset_name}")
        x, y = loaders[dataset_name]()
        if len(x) < 50:
            raise ValueError(f"Dataset {dataset_name} has too few rows.")
        datasets[dataset_name] = (x, y)
        print(f"  shape: X={x.shape}, y={y.shape}")

    return datasets


# ============================================================
# Outer split construction
# ============================================================


def make_random_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: VarianceFloorConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))

    if (
        cfg.max_samples_per_dataset is not None
        and len(indices) > cfg.max_samples_per_dataset
    ):
        indices = rng.choice(
            indices,
            size=cfg.max_samples_per_dataset,
            replace=False,
        )

    indices = rng.permutation(indices)
    n_used = len(indices)
    n_train = int(round(cfg.train_fraction * n_used))
    n_cal = int(round(cfg.cal_fraction * n_used))

    n_train = max(n_train, 30)
    n_cal = max(n_cal, 30)
    if n_train + n_cal >= n_used:
        n_train = int(0.60 * n_used)
        n_cal = int(0.20 * n_used)

    train_idx = indices[:n_train]
    cal_idx = indices[n_train : n_train + n_cal]
    test_idx = indices[n_train + n_cal :]
    if len(test_idx) < 25:
        raise ValueError("Test split is too small for support quintiles.")

    data = standardize_with_train(
        x_train=x[train_idx],
        x_cal=x[cal_idx],
        x_test=x[test_idx],
        y_train=y[train_idx],
        y_cal=y[cal_idx],
        y_test=y[test_idx],
    )
    data["n_train_actual"] = int(len(train_idx))
    data["n_cal_actual"] = int(len(cal_idx))
    data["n_test_actual"] = int(len(test_idx))
    return data


# ============================================================
# Heteroscedastic neural model
# ============================================================


class HeteroMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        variance_floor: float,
    ):
        super().__init__()
        self.variance_floor = float(variance_floor)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(feature_dim, 1)
        self.var_head = nn.Linear(feature_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.features(x)
        mu = self.mean_head(phi)
        base_h2 = F.softplus(self.var_head(phi))
        h2 = base_h2 + self.variance_floor
        return mu, h2, phi


def train_hetero_model(
    x: np.ndarray,
    y: np.ndarray,
    variance_floor: float,
    seed: int,
    cfg: VarianceFloorConfig,
) -> Tuple[HeteroMLP, Dict[str, float]]:
    # The fitting/validation split is identical across floors.
    fit_idx, validation_idx = deterministic_split(
        len(x),
        cfg.validation_fraction,
        seed=100_000 + seed,
    )

    # Architecture is identical across floors, so the initialization is also
    # identical. This isolates the effect of changing the floor as closely as
    # possible while still retraining the likelihood model.
    set_seed(200_000 + 1000 * seed)

    model = HeteroMLP(
        input_dim=x.shape[1],
        hidden_dim=cfg.hidden_dim,
        feature_dim=cfg.feature_dim,
        variance_floor=float(variance_floor),
    ).to(DEVICE)

    x_tensor = to_tensor(x)
    y_tensor = to_tensor(y)
    fit_idx_tensor = torch.tensor(fit_idx, dtype=torch.long, device=DEVICE)
    validation_idx_tensor = torch.tensor(
        validation_idx,
        dtype=torch.long,
        device=DEVICE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_state = None
    best_validation_loss = float("inf")
    wait = 0
    completed_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        completed_epochs = epoch + 1
        model.train()
        optimizer.zero_grad()

        mu, h2, _ = model(x_tensor[fit_idx_tensor])
        residual2 = (y_tensor[fit_idx_tensor] - mu) ** 2
        loss = 0.5 * torch.mean(residual2 / h2 + torch.log(h2))
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at floor={variance_floor:g}."
            )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_mu, validation_h2, _ = model(
                x_tensor[validation_idx_tensor]
            )
            validation_residual2 = (
                y_tensor[validation_idx_tensor] - validation_mu
            ) ** 2
            validation_loss = 0.5 * torch.mean(
                validation_residual2 / validation_h2
                + torch.log(validation_h2)
            ).item()

        if validation_loss < best_validation_loss - cfg.min_delta:
            best_validation_loss = float(validation_loss)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            break

    training_time_sec = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    diagnostics = {
        "training_time_sec": float(training_time_sec),
        "completed_epochs": int(completed_epochs),
        "best_validation_loss": float(best_validation_loss),
    }
    return model, diagnostics


def predict_hetero(
    model: HeteroMLP,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = to_tensor(x)
        phi = model.features(x_tensor)
        mu = model.mean_head(phi)
        base_h2 = F.softplus(model.var_head(phi))
        h2 = base_h2 + model.variance_floor

    return (
        to_numpy(mu).reshape(-1, 1),
        to_numpy(h2).reshape(-1, 1),
        to_numpy(base_h2).reshape(-1, 1),
        to_numpy(phi),
    )


def predict_all_splits(
    model: HeteroMLP,
    data: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    mu_train, h2_train, base_h2_train, phi_train = predict_hetero(
        model,
        data["x_train"],
    )
    mu_cal, h2_cal, base_h2_cal, phi_cal = predict_hetero(
        model,
        data["x_cal"],
    )
    mu_test, h2_test, base_h2_test, phi_test = predict_hetero(
        model,
        data["x_test"],
    )

    return {
        "mu_train": mu_train,
        "h2_train": h2_train,
        "base_h2_train": base_h2_train,
        "phi_train": phi_train,
        "mu_cal": mu_cal,
        "h2_cal": h2_cal,
        "base_h2_cal": base_h2_cal,
        "phi_cal": phi_cal,
        "mu_test": mu_test,
        "h2_test": h2_test,
        "base_h2_test": base_h2_test,
        "phi_test": phi_test,
    }


# ============================================================
# Last-layer geometry and prior-precision selection
# ============================================================


def add_bias_column(phi: np.ndarray) -> np.ndarray:
    ones = np.ones((phi.shape[0], 1), dtype=np.float64)
    return np.concatenate([phi.astype(np.float64), ones], axis=1)


def invert_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def build_weighted_precision(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    variance_floor: float,
    cfg: VarianceFloorConfig,
) -> np.ndarray:
    h_augmented = add_bias_column(phi_train)
    weights = 1.0 / np.maximum(
        h2_train.reshape(-1).astype(np.float64),
        float(variance_floor),
    )

    precision = float(prior_precision) * np.eye(
        h_augmented.shape[1],
        dtype=np.float64,
    )
    precision += h_augmented.T @ (h_augmented * weights[:, None])
    precision += cfg.jitter * np.eye(
        h_augmented.shape[1],
        dtype=np.float64,
    )
    return precision


def compute_weighted_laplace_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    variance_floor: float,
    cfg: VarianceFloorConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    precision = build_weighted_precision(
        phi_train=phi_train,
        h2_train=h2_train,
        prior_precision=prior_precision,
        variance_floor=variance_floor,
        cfg=cfg,
    )
    covariance = invert_positive_semidefinite(precision)
    return covariance, precision


def quadratic_form(
    phi: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    h_augmented = add_bias_column(phi)
    values = np.sum(
        (h_augmented @ covariance) * h_augmented,
        axis=1,
        keepdims=True,
    )
    return np.maximum(values, 0.0).astype(np.float32)


def select_prior_precision_training_only(
    dataset_name: str,
    seed: int,
    variance_floor: float,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    cfg: VarianceFloorConfig,
) -> Tuple[float, List[Dict[str, float]], float]:
    # The inner prior-selection split is identical across floors.
    fit_idx, validation_idx = deterministic_split(
        len(data["x_train"]),
        cfg.validation_fraction,
        seed=300_000 + seed,
    )

    y_validation = data["y_train_std"][validation_idx]
    mu_validation = predictions["mu_train"][validation_idx]
    h2_validation = predictions["h2_train"][validation_idx]
    phi_validation = predictions["phi_train"][validation_idx]

    trace_rows: List[Dict[str, float]] = []
    best_lambda = float(cfg.lambda_grid[0])
    best_nll = float("inf")
    start_time = time.perf_counter()

    for prior_precision in cfg.lambda_grid:
        covariance, _ = compute_weighted_laplace_covariance(
            phi_train=predictions["phi_train"][fit_idx],
            h2_train=predictions["h2_train"][fit_idx],
            prior_precision=float(prior_precision),
            variance_floor=float(variance_floor),
            cfg=cfg,
        )
        epistemic_validation = quadratic_form(
            phi_validation,
            covariance,
        )
        total_variance_validation = np.maximum(
            h2_validation + epistemic_validation,
            float(variance_floor),
        )
        residual2 = (y_validation - mu_validation) ** 2
        validation_nll = float(
            0.5
            * np.mean(
                residual2 / total_variance_validation
                + np.log(total_variance_validation)
            )
        )

        trace_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "variance_floor": float(variance_floor),
                "prior_precision": float(prior_precision),
                "validation_nll": validation_nll,
            }
        )

        if validation_nll < best_nll:
            best_nll = validation_nll
            best_lambda = float(prior_precision)

    elapsed_sec = time.perf_counter() - start_time
    for row in trace_rows:
        row["nll_excess_over_best"] = row["validation_nll"] - best_nll
        row["is_selected"] = bool(
            np.isclose(row["prior_precision"], best_lambda)
        )

    return best_lambda, trace_rows, float(elapsed_sec)


# ============================================================
# Independent learned-feature support groups
# ============================================================


def standardize_feature_space(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0, keepdims=True)
    scale = np.std(train_features, axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (
        ((train_features - mean) / scale).astype(np.float32),
        ((test_features - mean) / scale).astype(np.float32),
    )


def knn_support_distance(
    train_features: np.ndarray,
    test_features: np.ndarray,
    k: int,
) -> np.ndarray:
    n_neighbors = min(max(int(k), 1), len(train_features))
    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )
    nearest_neighbors.fit(train_features)
    distances, _ = nearest_neighbors.kneighbors(
        test_features,
        return_distance=True,
    )
    return np.mean(distances, axis=1)


def rank_based_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    groups = np.empty(len(values), dtype=int)
    groups[order] = (
        np.floor(np.arange(len(values)) * n_groups / len(values)).astype(int)
        + 1
    )
    return np.minimum(groups, n_groups)


def learned_feature_support_quintiles(
    phi_train: np.ndarray,
    phi_test: np.ndarray,
    cfg: VarianceFloorConfig,
) -> np.ndarray:
    phi_train_standardized, phi_test_standardized = standardize_feature_space(
        phi_train,
        phi_test,
    )
    distances = knn_support_distance(
        phi_train_standardized,
        phi_test_standardized,
        cfg.support_k,
    )
    return rank_based_groups(distances, cfg.support_quintiles)


# ============================================================
# Conformal interval and metric helpers
# ============================================================


def interval_from_scale(
    mu_cal: np.ndarray,
    scale_cal: np.ndarray,
    mu_test: np.ndarray,
    scale_test: np.ndarray,
    data: Mapping[str, np.ndarray],
    cfg: VarianceFloorConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scale_cal = np.maximum(scale_cal, 1e-12)
    scale_test = np.maximum(scale_test, 1e-12)

    scores = np.abs(data["y_cal_std"] - mu_cal) / scale_cal
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_standardized = mu_test - qhat * scale_test
    upper_standardized = mu_test + qhat * scale_test
    lower = unstandardize_y(
        lower_standardized,
        float(data["y_mean"]),
        float(data["y_scale"]),
    )
    upper = unstandardize_y(
        upper_standardized,
        float(data["y_mean"]),
        float(data["y_scale"]),
    )
    return lower, upper, float(qhat)


def compute_interval_metrics(
    dataset_name: str,
    seed: int,
    variance_floor: float,
    method_name: str,
    data: Mapping[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float,
    support_quintiles: np.ndarray,
    cfg: VarianceFloorConfig,
) -> Dict[str, float]:
    y = np.asarray(data["y_test"]).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    covered = (y >= lower) & (y <= upper)
    widths = upper - lower
    scores = interval_score(y, lower, upper, cfg.alpha)
    y_scale = max(float(data["y_scale"]), 1e-12)

    q5_mask = support_quintiles == cfg.support_quintiles
    if not np.any(q5_mask):
        raise RuntimeError("Weakest support quintile is empty.")

    return {
        "dataset": dataset_name,
        "seed": int(seed),
        "variance_floor": float(variance_floor),
        "method": method_name,
        "marginal_coverage": float(np.mean(covered)),
        "normalized_width": float(np.mean(widths) / y_scale),
        "normalized_interval_score": float(np.mean(scores) / y_scale),
        "q5_coverage": float(np.mean(covered[q5_mask])),
        "q5_normalized_width": float(np.mean(widths[q5_mask]) / y_scale),
        "q5_normalized_interval_score": float(
            np.mean(scores[q5_mask]) / y_scale
        ),
        "qhat": float(qhat),
    }


def run_lacp_from_predictions(
    dataset_name: str,
    seed: int,
    variance_floor: float,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    support_quintiles: np.ndarray,
    cfg: VarianceFloorConfig,
) -> Dict[str, float]:
    scale_cal = np.sqrt(
        np.maximum(predictions["h2_cal"], float(variance_floor))
    )
    scale_test = np.sqrt(
        np.maximum(predictions["h2_test"], float(variance_floor))
    )
    lower, upper, qhat = interval_from_scale(
        predictions["mu_cal"],
        scale_cal,
        predictions["mu_test"],
        scale_test,
        data,
        cfg,
    )
    row = compute_interval_metrics(
        dataset_name,
        seed,
        variance_floor,
        "LACP",
        data,
        lower,
        upper,
        qhat,
        support_quintiles,
        cfg,
    )
    row.update(
        {
            "selected_prior_precision": np.nan,
            "epistemic_fraction": np.nan,
            "q5_epistemic_fraction": np.nan,
            "scale_ratio_to_aleatoric": 1.0,
            "q5_scale_ratio_to_aleatoric": 1.0,
        }
    )
    return row


def run_claps_from_predictions(
    dataset_name: str,
    seed: int,
    variance_floor: float,
    selected_prior_precision: float,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    support_quintiles: np.ndarray,
    cfg: VarianceFloorConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    covariance, precision = compute_weighted_laplace_covariance(
        phi_train=predictions["phi_train"],
        h2_train=predictions["h2_train"],
        prior_precision=selected_prior_precision,
        variance_floor=variance_floor,
        cfg=cfg,
    )

    epistemic_cal = quadratic_form(predictions["phi_cal"], covariance)
    epistemic_test = quadratic_form(predictions["phi_test"], covariance)
    total_variance_cal = np.maximum(
        predictions["h2_cal"] + epistemic_cal,
        float(variance_floor),
    )
    total_variance_test = np.maximum(
        predictions["h2_test"] + epistemic_test,
        float(variance_floor),
    )

    lower, upper, qhat = interval_from_scale(
        predictions["mu_cal"],
        np.sqrt(total_variance_cal),
        predictions["mu_test"],
        np.sqrt(total_variance_test),
        data,
        cfg,
    )
    row = compute_interval_metrics(
        dataset_name,
        seed,
        variance_floor,
        "CLAPS",
        data,
        lower,
        upper,
        qhat,
        support_quintiles,
        cfg,
    )

    q5_mask = support_quintiles == cfg.support_quintiles
    epistemic_fraction_pointwise = epistemic_test / np.maximum(
        total_variance_test,
        float(variance_floor),
    )
    scale_ratio_pointwise = np.sqrt(total_variance_test) / np.sqrt(
        np.maximum(predictions["h2_test"], float(variance_floor))
    )

    row.update(
        {
            "selected_prior_precision": float(selected_prior_precision),
            "epistemic_fraction": float(
                np.mean(epistemic_fraction_pointwise)
            ),
            "q5_epistemic_fraction": float(
                np.mean(epistemic_fraction_pointwise[q5_mask])
            ),
            "scale_ratio_to_aleatoric": float(
                np.mean(scale_ratio_pointwise)
            ),
            "q5_scale_ratio_to_aleatoric": float(
                np.mean(scale_ratio_pointwise[q5_mask])
            ),
        }
    )

    weights = 1.0 / np.maximum(
        predictions["h2_train"].reshape(-1),
        float(variance_floor),
    )
    floor_fraction_train = float(variance_floor) / predictions[
        "h2_train"
    ].reshape(-1)
    floor_fraction_test = float(variance_floor) / predictions[
        "h2_test"
    ].reshape(-1)

    # "Floor dominant" means the additive floor contributes at least half of
    # the final aleatoric variance, equivalent to base_h2 <= variance_floor.
    floor_dominant_train = (
        predictions["base_h2_train"].reshape(-1) <= float(variance_floor)
    )
    floor_dominant_test = (
        predictions["base_h2_test"].reshape(-1) <= float(variance_floor)
    )

    numerical_diagnostics = {
        "h2_train_min": float(np.min(predictions["h2_train"])),
        "h2_train_p01": float(np.quantile(predictions["h2_train"], 0.01)),
        "h2_test_min": float(np.min(predictions["h2_test"])),
        "h2_test_p01": float(np.quantile(predictions["h2_test"], 0.01)),
        "floor_contribution_mean_train": float(
            np.mean(floor_fraction_train)
        ),
        "floor_contribution_mean_test": float(
            np.mean(floor_fraction_test)
        ),
        "floor_dominant_rate_train": float(
            np.mean(floor_dominant_train)
        ),
        "floor_dominant_rate_test": float(
            np.mean(floor_dominant_test)
        ),
        "precision_weight_p99": float(np.quantile(weights, 0.99)),
        "precision_weight_max": float(np.max(weights)),
        "precision_condition_number": float(np.linalg.cond(precision)),
        "log10_precision_condition_number": float(
            np.log10(max(np.linalg.cond(precision), 1.0))
        ),
    }
    return row, numerical_diagnostics


# ============================================================
# One dataset-seed-floor run
# ============================================================


def run_single_floor(
    dataset_name: str,
    data: Mapping[str, np.ndarray],
    seed: int,
    variance_floor: float,
    cfg: VarianceFloorConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    model, training_diagnostics = train_hetero_model(
        x=data["x_train"],
        y=data["y_train_std"],
        variance_floor=variance_floor,
        seed=seed,
        cfg=cfg,
    )
    predictions = predict_all_splits(model, data)

    support_quintiles = learned_feature_support_quintiles(
        predictions["phi_train"],
        predictions["phi_test"],
        cfg,
    )

    selected_lambda, selection_trace, selection_time_sec = (
        select_prior_precision_training_only(
            dataset_name=dataset_name,
            seed=seed,
            variance_floor=variance_floor,
            data=data,
            predictions=predictions,
            cfg=cfg,
        )
    )

    lacp_row = run_lacp_from_predictions(
        dataset_name=dataset_name,
        seed=seed,
        variance_floor=variance_floor,
        data=data,
        predictions=predictions,
        support_quintiles=support_quintiles,
        cfg=cfg,
    )
    claps_row, numerical_diagnostics = run_claps_from_predictions(
        dataset_name=dataset_name,
        seed=seed,
        variance_floor=variance_floor,
        selected_prior_precision=selected_lambda,
        data=data,
        predictions=predictions,
        support_quintiles=support_quintiles,
        cfg=cfg,
    )

    shared_diagnostics = {
        "n_train": int(data["n_train_actual"]),
        "n_cal": int(data["n_cal_actual"]),
        "n_test": int(data["n_test_actual"]),
        "training_time_sec": training_diagnostics["training_time_sec"],
        "completed_epochs": training_diagnostics["completed_epochs"],
        "best_validation_loss": training_diagnostics[
            "best_validation_loss"
        ],
        "lambda_selection_time_sec": float(selection_time_sec),
    }
    lacp_row.update(shared_diagnostics)
    claps_row.update(shared_diagnostics)
    claps_row.update(numerical_diagnostics)

    # Store the same variance-head diagnostics on LACP so raw paired data are
    # self-contained, while last-layer-only diagnostics remain unavailable.
    for key in [
        "h2_train_min",
        "h2_train_p01",
        "h2_test_min",
        "h2_test_p01",
        "floor_contribution_mean_train",
        "floor_contribution_mean_test",
        "floor_dominant_rate_train",
        "floor_dominant_rate_test",
    ]:
        lacp_row[key] = numerical_diagnostics[key]
    for key in [
        "precision_weight_p99",
        "precision_weight_max",
        "precision_condition_number",
        "log10_precision_condition_number",
    ]:
        lacp_row[key] = np.nan

    del model, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [lacp_row, claps_row], selection_trace


# ============================================================
# Experiment loop
# ============================================================


def run_experiment(
    cfg: VarianceFloorConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not any(
        np.isclose(cfg.reference_variance_floor, floor)
        for floor in cfg.variance_floor_grid
    ):
        raise ValueError(
            "reference_variance_floor must be in variance_floor_grid."
        )

    datasets = load_all_datasets(cfg)
    result_rows: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, float]] = []

    total_jobs = (
        len(datasets) * cfg.num_seeds * len(cfg.variance_floor_grid)
    )
    completed_jobs = 0
    start_time = time.time()

    for dataset_name, (x, y) in datasets.items():
        for seed in range(cfg.num_seeds):
            set_seed(seed)
            data = make_random_split(x, y, seed, cfg)

            for variance_floor in cfg.variance_floor_grid:
                completed_jobs += 1
                if cfg.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"[{completed_jobs}/{total_jobs}] "
                        f"Dataset={dataset_name} | Seed={seed} | "
                        f"Floor={variance_floor:g} | "
                        f"Elapsed={elapsed:.1f}s"
                    )

                floor_rows, floor_selection_rows = run_single_floor(
                    dataset_name=dataset_name,
                    data=data,
                    seed=seed,
                    variance_floor=float(variance_floor),
                    cfg=cfg,
                )
                result_rows.extend(floor_rows)
                selection_rows.extend(floor_selection_rows)

    return pd.DataFrame(result_rows), pd.DataFrame(selection_rows)


# ============================================================
# Derived paired effects
# ============================================================


def add_lacp_relative_effects(results: pd.DataFrame) -> pd.DataFrame:
    lacp = (
        results[results["method"] == "LACP"]
        [
            [
                "dataset",
                "seed",
                "variance_floor",
                "marginal_coverage",
                "normalized_width",
                "normalized_interval_score",
                "q5_coverage",
                "q5_normalized_width",
                "q5_normalized_interval_score",
                "qhat",
            ]
        ]
        .rename(
            columns={
                "marginal_coverage": "lacp_coverage",
                "normalized_width": "lacp_width",
                "normalized_interval_score": "lacp_score",
                "q5_coverage": "lacp_q5_coverage",
                "q5_normalized_width": "lacp_q5_width",
                "q5_normalized_interval_score": "lacp_q5_score",
                "qhat": "lacp_qhat",
            }
        )
    )

    output = results.merge(
        lacp,
        on=["dataset", "seed", "variance_floor"],
        how="left",
    )
    output["coverage_diff_vs_lacp"] = (
        output["marginal_coverage"] - output["lacp_coverage"]
    )
    output["width_change_vs_lacp"] = (
        output["normalized_width"] / output["lacp_width"] - 1.0
    )
    output["score_change_vs_lacp"] = (
        output["normalized_interval_score"] / output["lacp_score"] - 1.0
    )
    output["q5_coverage_diff_vs_lacp"] = (
        output["q5_coverage"] - output["lacp_q5_coverage"]
    )
    output["q5_width_change_vs_lacp"] = (
        output["q5_normalized_width"] / output["lacp_q5_width"] - 1.0
    )
    output["q5_score_change_vs_lacp"] = (
        output["q5_normalized_interval_score"] / output["lacp_q5_score"]
        - 1.0
    )
    output["qhat_ratio_to_lacp"] = output["qhat"] / output["lacp_qhat"]
    return output


def add_reference_floor_effects(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    reference = (
        claps[
            np.isclose(
                claps["variance_floor"],
                cfg.reference_variance_floor,
            )
        ]
        [
            [
                "dataset",
                "seed",
                "marginal_coverage",
                "normalized_width",
                "normalized_interval_score",
            ]
        ]
        .rename(
            columns={
                "marginal_coverage": "reference_floor_coverage",
                "normalized_width": "reference_floor_width",
                "normalized_interval_score": "reference_floor_score",
            }
        )
    )

    output = results.merge(reference, on=["dataset", "seed"], how="left")
    output["coverage_diff_vs_reference_floor"] = (
        output["marginal_coverage"] - output["reference_floor_coverage"]
    )
    output["width_change_vs_reference_floor"] = (
        output["normalized_width"] / output["reference_floor_width"] - 1.0
    )
    output["score_change_vs_reference_floor"] = (
        output["normalized_interval_score"] / output["reference_floor_score"]
        - 1.0
    )
    return output


# ============================================================
# Summary tables
# ============================================================


PERFORMANCE_METRICS: Tuple[Tuple[str, str, float, str, int], ...] = (
    ("marginal_coverage", "Coverage", 1.0, "", 4),
    ("coverage_diff_vs_lacp", "Coverage Delta", 1.0, "", 4),
    ("width_change_vs_lacp", "Width Delta vs LACP", 100.0, "%", 2),
    ("score_change_vs_lacp", "Score Delta vs LACP", 100.0, "%", 2),
    (
        "q5_coverage_diff_vs_lacp",
        "Weak-Q5 Coverage Delta",
        1.0,
        "",
        4,
    ),
    (
        "q5_width_change_vs_lacp",
        "Weak-Q5 Width Delta",
        100.0,
        "%",
        2,
    ),
    (
        "q5_score_change_vs_lacp",
        "Weak-Q5 Score Delta",
        100.0,
        "%",
        2,
    ),
    (
        "width_change_vs_reference_floor",
        "CLAPS Width Delta vs Ref. Floor",
        100.0,
        "%",
        2,
    ),
    (
        "score_change_vs_reference_floor",
        "CLAPS Score Delta vs Ref. Floor",
        100.0,
        "%",
        2,
    ),
    ("epistemic_fraction", "Epistemic Fraction", 100.0, "%", 2),
    (
        "q5_epistemic_fraction",
        "Weak-Q5 Epistemic Fraction",
        100.0,
        "%",
        2,
    ),
)


def summarize_performance_by_floor(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    rows: List[Dict[str, str]] = []

    for variance_floor in cfg.variance_floor_grid:
        subset = claps[
            np.isclose(claps["variance_floor"], variance_floor)
        ]
        if subset.empty:
            continue

        row: Dict[str, str] = {
            "Variance Floor": f"{float(variance_floor):g}",
            "Selected Lambda": format_mean_se(
                float(np.mean(subset["selected_prior_precision"])),
                safe_sem(subset["selected_prior_precision"]),
                digits=cfg.summary_digits,
            ),
        }
        for metric, label, multiplier, suffix, digits in PERFORMANCE_METRICS:
            values = subset[metric].to_numpy(dtype=float)
            row[label] = format_mean_se(
                float(np.nanmean(values)),
                safe_sem(values),
                digits=digits,
                multiplier=multiplier,
                suffix=suffix,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_numerics_by_floor(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    metric_specs = (
        ("h2_test_min", "Min h2", 1.0, "", 6),
        ("h2_test_p01", "P01 h2", 1.0, "", 6),
        (
            "floor_contribution_mean_test",
            "Mean Floor Contribution",
            100.0,
            "%",
            3,
        ),
        (
            "floor_dominant_rate_test",
            "Floor-Dominant Rate",
            100.0,
            "%",
            3,
        ),
        ("precision_weight_p99", "P99 Precision Weight", 1.0, "", 3),
        ("precision_weight_max", "Max Precision Weight", 1.0, "", 3),
        (
            "log10_precision_condition_number",
            "log10 Condition Number",
            1.0,
            "",
            3,
        ),
        (
            "scale_ratio_to_aleatoric",
            "Scale Ratio",
            1.0,
            "",
            4,
        ),
        ("training_time_sec", "Training Time", 1.0, " s", 2),
    )

    rows: List[Dict[str, str]] = []
    for variance_floor in cfg.variance_floor_grid:
        subset = claps[
            np.isclose(claps["variance_floor"], variance_floor)
        ]
        if subset.empty:
            continue

        row: Dict[str, str] = {
            "Variance Floor": f"{float(variance_floor):g}",
        }
        for metric, label, multiplier, suffix, digits in metric_specs:
            values = subset[metric].to_numpy(dtype=float)
            row[label] = format_mean_se(
                float(np.nanmean(values)),
                safe_sem(values),
                digits=digits,
                multiplier=multiplier,
                suffix=suffix,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_dataset_floor(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    compact_metrics = (
        ("marginal_coverage", "Coverage", 1.0, "", 4),
        ("width_change_vs_lacp", "Width Delta vs LACP", 100.0, "%", 2),
        ("score_change_vs_lacp", "Score Delta vs LACP", 100.0, "%", 2),
        (
            "q5_score_change_vs_lacp",
            "Weak-Q5 Score Delta",
            100.0,
            "%",
            2,
        ),
        ("epistemic_fraction", "Epistemic Fraction", 100.0, "%", 2),
        (
            "floor_contribution_mean_test",
            "Floor Contribution",
            100.0,
            "%",
            3,
        ),
        (
            "log10_precision_condition_number",
            "log10 Condition Number",
            1.0,
            "",
            3,
        ),
    )

    rows: List[Dict[str, str]] = []
    for dataset_name in cfg.dataset_names:
        for variance_floor in cfg.variance_floor_grid:
            subset = claps[
                (claps["dataset"] == dataset_name)
                & np.isclose(claps["variance_floor"], variance_floor)
            ]
            if subset.empty:
                continue

            row: Dict[str, str] = {
                "Dataset": dataset_name,
                "Variance Floor": f"{float(variance_floor):g}",
            }
            for metric, label, multiplier, suffix, digits in compact_metrics:
                values = subset[metric].to_numpy(dtype=float)
                row[label] = format_mean_se(
                    float(np.nanmean(values)),
                    safe_sem(values),
                    digits=digits,
                    multiplier=multiplier,
                    suffix=suffix,
                )
            rows.append(row)

    return pd.DataFrame(rows)


def build_lambda_selection_frequency(
    selection_trace: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    selected = (
        selection_trace[selection_trace["is_selected"]]
        [["dataset", "seed", "variance_floor", "prior_precision"]]
        .drop_duplicates()
    )

    rows = []
    for variance_floor in cfg.variance_floor_grid:
        floor_selected = selected[
            np.isclose(selected["variance_floor"], variance_floor)
        ]
        denominator = len(floor_selected)
        for prior_precision in cfg.lambda_grid:
            count = int(
                np.sum(
                    np.isclose(
                        floor_selected["prior_precision"],
                        prior_precision,
                    )
                )
            )
            rows.append(
                {
                    "Variance Floor": f"{float(variance_floor):g}",
                    "Prior Precision": f"{float(prior_precision):g}",
                    "Count": count,
                    "Selection Rate": count / max(denominator, 1),
                }
            )

    output = pd.DataFrame(rows)
    output["Selection Rate"] = output["Selection Rate"].map(
        lambda value: f"{100.0 * value:.2f}%"
    )
    return output


# ============================================================
# Hierarchical paired bootstrap
# ============================================================


def hierarchical_bootstrap_mean(
    data: pd.DataFrame,
    metric: str,
    cfg: VarianceFloorConfig,
    seed_offset: int,
) -> Tuple[float, float, float]:
    datasets = data["dataset"].unique().tolist()
    if not datasets:
        raise ValueError("No datasets are available for bootstrap.")

    observed = float(np.mean(data[metric].to_numpy(dtype=float)))
    rng = np.random.default_rng(cfg.bootstrap_seed + seed_offset)
    bootstrap_means = np.empty(cfg.bootstrap_repeats, dtype=float)

    dataset_values = {
        dataset_name: data.loc[
            data["dataset"] == dataset_name,
            metric,
        ].to_numpy(dtype=float)
        for dataset_name in datasets
    }

    for bootstrap_index in range(cfg.bootstrap_repeats):
        sampled_datasets = rng.choice(
            datasets,
            size=len(datasets),
            replace=True,
        )
        sampled_values: List[float] = []
        for dataset_name in sampled_datasets:
            values = dataset_values[dataset_name]
            sampled_values.extend(
                rng.choice(values, size=len(values), replace=True).tolist()
            )
        bootstrap_means[bootstrap_index] = float(np.mean(sampled_values))

    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return observed, float(ci_low), float(ci_high)


def build_floor_bootstrap_table(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    metric_specs = (
        ("coverage_diff_vs_lacp", "Coverage Delta", 1.0, ""),
        ("width_change_vs_lacp", "Width Delta vs LACP", 100.0, "%"),
        ("score_change_vs_lacp", "Score Delta vs LACP", 100.0, "%"),
        (
            "q5_coverage_diff_vs_lacp",
            "Weak-Q5 Coverage Delta",
            1.0,
            "",
        ),
        (
            "q5_width_change_vs_lacp",
            "Weak-Q5 Width Delta",
            100.0,
            "%",
        ),
        (
            "q5_score_change_vs_lacp",
            "Weak-Q5 Score Delta",
            100.0,
            "%",
        ),
    )

    rows = []
    for floor_index, variance_floor in enumerate(cfg.variance_floor_grid):
        subset = claps[
            np.isclose(claps["variance_floor"], variance_floor)
        ]
        if subset.empty:
            continue

        for metric_index, (metric, label, multiplier, suffix) in enumerate(
            metric_specs
        ):
            mean_value, ci_low, ci_high = hierarchical_bootstrap_mean(
                subset,
                metric,
                cfg,
                seed_offset=1000 * floor_index + metric_index,
            )
            rows.append(
                {
                    "Variance Floor": f"{float(variance_floor):g}",
                    "Effect": label,
                    "Mean": f"{multiplier * mean_value:.4f}{suffix}",
                    "95% CI": (
                        f"[{multiplier * ci_low:.4f}, "
                        f"{multiplier * ci_high:.4f}]{suffix}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_reference_floor_bootstrap_table(
    results: pd.DataFrame,
    cfg: VarianceFloorConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    metric_specs = (
        (
            "coverage_diff_vs_reference_floor",
            "Coverage Delta vs Reference Floor",
            1.0,
            "",
        ),
        (
            "width_change_vs_reference_floor",
            "Width Delta vs Reference Floor",
            100.0,
            "%",
        ),
        (
            "score_change_vs_reference_floor",
            "Score Delta vs Reference Floor",
            100.0,
            "%",
        ),
    )

    rows = []
    for floor_index, variance_floor in enumerate(cfg.variance_floor_grid):
        subset = claps[
            np.isclose(claps["variance_floor"], variance_floor)
        ]
        if subset.empty:
            continue

        for metric_index, (metric, label, multiplier, suffix) in enumerate(
            metric_specs
        ):
            mean_value, ci_low, ci_high = hierarchical_bootstrap_mean(
                subset,
                metric,
                cfg,
                seed_offset=20_000 + 1000 * floor_index + metric_index,
            )
            rows.append(
                {
                    "Variance Floor": f"{float(variance_floor):g}",
                    "Effect": label,
                    "Mean": f"{multiplier * mean_value:.4f}{suffix}",
                    "95% CI": (
                        f"[{multiplier * ci_low:.4f}, "
                        f"{multiplier * ci_high:.4f}]{suffix}"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Saving and main entry point
# ============================================================


def save_outputs(
    output_frames: Mapping[str, pd.DataFrame],
    cfg: VarianceFloorConfig,
) -> None:
    if not cfg.save_csv:
        return

    for suffix, frame in output_frames.items():
        path = f"{cfg.output_prefix}_{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")


def main(
    cfg: VarianceFloorConfig = CFG,
) -> Dict[str, pd.DataFrame]:
    prepare_packages()
    results_df, selection_trace_df = run_experiment(cfg)
    results_df = add_lacp_relative_effects(results_df)
    results_df = add_reference_floor_effects(results_df, cfg)

    floor_performance_summary_df = summarize_performance_by_floor(
        results_df,
        cfg,
    )
    floor_numerical_summary_df = summarize_numerics_by_floor(
        results_df,
        cfg,
    )
    dataset_floor_summary_df = summarize_by_dataset_floor(results_df, cfg)
    lambda_selection_frequency_df = build_lambda_selection_frequency(
        selection_trace_df,
        cfg,
    )
    floor_bootstrap_df = build_floor_bootstrap_table(results_df, cfg)
    reference_floor_bootstrap_df = build_reference_floor_bootstrap_table(
        results_df,
        cfg,
    )

    output_frames = {
        "raw_results": results_df,
        "selection_trace": selection_trace_df,
        "performance_summary": floor_performance_summary_df,
        "numerical_summary": floor_numerical_summary_df,
        "dataset_floor_summary": dataset_floor_summary_df,
        "lambda_selection_frequency": lambda_selection_frequency_df,
        "floor_bootstrap": floor_bootstrap_df,
        "reference_floor_bootstrap": reference_floor_bootstrap_df,
    }
    save_outputs(output_frames, cfg)

    print("\nFinished variance-floor sensitivity analysis.")

    print("\nPerformance sensitivity: mean ± standard error")
    display(floor_performance_summary_df)

    print("\nVariance-head and last-layer numerical diagnostics")
    display(floor_numerical_summary_df)

    print("\nTraining-only prior-precision selection frequency")
    display(lambda_selection_frequency_df)

    print("\nHierarchical paired bootstrap: CLAPS versus matched LACP")
    display(floor_bootstrap_df)

    print(
        "\nHierarchical paired bootstrap: CLAPS versus reference "
        f"floor {cfg.reference_variance_floor:g}"
    )
    display(reference_floor_bootstrap_df)

    print("\nDataset-level sensitivity")
    for dataset_name in cfg.dataset_names:
        print(f"\nDataset: {dataset_name}")
        display(
            dataset_floor_summary_df[
                dataset_floor_summary_df["Dataset"] == dataset_name
            ]
        )

    if cfg.display_raw_results:
        print("\nRaw per-run results")
        display(results_df.round(cfg.summary_digits))

        print("\nPrior-selection trace")
        display(selection_trace_df.round(cfg.summary_digits))

    return output_frames


if __name__ == "__main__":
    OUTPUTS = main(CFG)
    results_df = OUTPUTS["raw_results"]
    selection_trace_df = OUTPUTS["selection_trace"]
    floor_performance_summary_df = OUTPUTS["performance_summary"]
    floor_numerical_summary_df = OUTPUTS["numerical_summary"]
    dataset_floor_summary_df = OUTPUTS["dataset_floor_summary"]
    lambda_selection_frequency_df = OUTPUTS[
        "lambda_selection_frequency"
    ]
    floor_bootstrap_df = OUTPUTS["floor_bootstrap"]
    reference_floor_bootstrap_df = OUTPUTS[
        "reference_floor_bootstrap"
    ]


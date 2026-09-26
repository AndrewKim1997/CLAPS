
# -*- coding: utf-8 -*-
"""
Appendix: Representation-Dimension Sensitivity for CLAPS
=========================================================

Standalone Google Colab / Python implementation aligned with the revised
real-data benchmark and the revised prior-precision selection protocol.

Purpose
-------
This experiment varies the learned last-layer representation dimension while
keeping the outer data split, training protocol, and matched LACP comparison
fixed. Because changing the representation dimension changes the neural
regressor itself, a separate heteroscedastic model is trained for each
(dataset, seed, feature dimension) configuration.

Protocol
--------
1. Use representative real-data datasets with the same 60/20/20
   train/calibration/test split and training-only standardization as revised
   Experiment 4.
2. For each feature dimension, fit one heteroscedastic neural regressor.
3. Select the CLAPS prior precision from a training-only inner split by
   predictive Gaussian negative log-likelihood. Calibration and test labels
   are never used for prior-precision selection.
4. Construct LACP and CLAPS intervals from the same fitted regressor.
5. Report CLAPS-minus-LACP effects overall and in the weakest learned-feature
   support quintile.
6. Quantify the incremental last-layer cost as representation dimension grows:
   covariance construction time, quadratic-form latency, and covariance
   storage.

Primary outputs
---------------
results_df
    Per-dataset, per-seed, per-dimension LACP and CLAPS results.
selection_trace_df
    Training-only validation NLL for every prior-precision candidate.
dimension_summary_df
    Aggregate CLAPS sensitivity summary relative to the matched LACP model.
dataset_dimension_summary_df
    Dataset-specific sensitivity summary.
computation_summary_df
    Training and last-layer computational-cost summary by feature dimension.
lambda_selection_frequency_df
    Prior-precision selection frequency by feature dimension.
dimension_bootstrap_df
    Hierarchical paired bootstrap intervals for CLAPS-minus-LACP effects at
    every feature dimension.

Notes
-----
The default grid is (16, 32, 64, 128). Add 256 to ``feature_dim_grid`` when a
larger-dimension stress test is desired. The default datasets match the former
appendix subset to keep the number of neural-network fits manageable; the full
eight-dataset benchmark can be enabled directly in the configuration.
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
class RepresentationDimensionConfig:
    # Representative subset retained from the original appendix.
    # For the complete benchmark, replace this tuple with all eight names:
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

    # Representation-dimension grid. Add 256 for a larger stress test.
    hidden_dim: int = 64
    feature_dim_grid: Tuple[int, ...] = (16, 32, 64, 128)

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

    # Last-layer Laplace.
    variance_floor: float = 1e-4
    jitter: float = 1e-6

    # Independent learned-feature support proxy.
    support_k: int = 20
    support_quintiles: int = 5

    # Timing. The final covariance is built once; the test quadratic form is
    # repeated only to stabilize the latency estimate.
    timing_repeats: int = 30

    # Hierarchical paired bootstrap.
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 20260801

    # Display and saving.
    verbose: bool = True
    summary_digits: int = 4
    save_csv: bool = True
    output_prefix: str = "representation_dimension"
    display_raw_results: bool = False


CFG = RepresentationDimensionConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("Appendix experiment: representation-dimension sensitivity")
print(f"Datasets: {CFG.dataset_names}")
print(f"Feature dimensions: {CFG.feature_dim_grid}")
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


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    n_val = max(1, int(round(n * validation_fraction)))
    n_val = min(n_val, n - 1)
    val_idx = indices[:n_val]
    fit_idx = indices[n_val:]
    return fit_idx, val_idx


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
    y_std: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    return y_std * y_scale + y_mean


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


def format_mean_se(
    mean_value: float,
    se_value: float,
    digits: int,
    multiplier: float = 1.0,
    suffix: str = "",
) -> str:
    mean_value *= multiplier
    se_value *= multiplier
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}{suffix}"


def safe_sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


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

    frame = x_df.copy()
    frame["_target"] = y_series.to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    y = frame["_target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x = frame.drop(columns=["_target"]).to_numpy(dtype=np.float32)
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

    # Columns 0--15 are features; column 17 is GT turbine decay coefficient.
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
        for column in ["cnt", "casual", "registered", "instant", "dteday"]
        if column in frame.columns
    ]
    return clean_xy(frame.drop(columns=drop_columns), target)


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing(as_frame=True)
    return clean_xy(data.data, data.target)


def load_all_datasets(
    cfg: RepresentationDimensionConfig,
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
    for name in cfg.dataset_names:
        if name not in loaders:
            raise ValueError(f"Unknown dataset name: {name}")

        print(f"Loading dataset: {name}")
        x, y = loaders[name]()
        if len(x) < 50:
            raise ValueError(f"Dataset {name} has too few usable rows.")

        datasets[name] = (x, y)
        print(f"  shape: X={x.shape}, y={y.shape}")

    return datasets


# ============================================================
# Train / calibration / test split
# ============================================================


def make_random_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: RepresentationDimensionConfig,
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
        self.variance_floor = variance_floor
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
        h2 = F.softplus(self.var_head(phi)) + self.variance_floor
        return mu, h2, phi


def train_hetero_model(
    x: np.ndarray,
    y: np.ndarray,
    feature_dim: int,
    seed: int,
    cfg: RepresentationDimensionConfig,
) -> Tuple[HeteroMLP, Dict[str, float]]:
    # The same deterministic fitting/validation split is used across feature
    # dimensions within a dataset-seed run.
    fit_idx, val_idx = deterministic_split(
        len(x),
        cfg.validation_fraction,
        seed=100_000 + seed,
    )

    # Reinitialize deterministically for each dimension.
    set_seed(200_000 + 1000 * seed + int(feature_dim))

    model = HeteroMLP(
        input_dim=x.shape[1],
        hidden_dim=cfg.hidden_dim,
        feature_dim=int(feature_dim),
        variance_floor=cfg.variance_floor,
    ).to(DEVICE)

    x_tensor = to_tensor(x)
    y_tensor = to_tensor(y)
    fit_idx_tensor = torch.tensor(fit_idx, dtype=torch.long, device=DEVICE)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_state = None
    best_validation_loss = float("inf")
    wait = 0
    completed_epochs = 0

    synchronize_device()
    start_time = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        completed_epochs = epoch + 1
        model.train()
        optimizer.zero_grad()

        mu, h2, _ = model(x_tensor[fit_idx_tensor])
        residual2 = (y_tensor[fit_idx_tensor] - mu) ** 2
        loss = 0.5 * torch.mean(residual2 / h2 + torch.log(h2))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite heteroscedastic training loss.")

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_mu, validation_h2, _ = model(x_tensor[val_idx_tensor])
            validation_residual2 = (
                y_tensor[val_idx_tensor] - validation_mu
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

    synchronize_device()
    training_time_sec = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    diagnostics = {
        "training_time_sec": float(training_time_sec),
        "completed_epochs": int(completed_epochs),
        "best_validation_loss": float(best_validation_loss),
        "parameter_count": int(count_parameters(model)),
    }
    return model, diagnostics


def predict_hetero(
    model: HeteroMLP,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        mu, h2, phi = model(to_tensor(x))
    return (
        to_numpy(mu).reshape(-1, 1),
        to_numpy(h2).reshape(-1, 1),
        to_numpy(phi),
    )


def predict_all_splits(
    model: HeteroMLP,
    data: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], float]:
    synchronize_device()
    start_time = time.perf_counter()

    mu_train, h2_train, phi_train = predict_hetero(
        model,
        data["x_train"],
    )
    mu_cal, h2_cal, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, phi_test = predict_hetero(model, data["x_test"])

    synchronize_device()
    elapsed = time.perf_counter() - start_time

    predictions = {
        "mu_train": mu_train,
        "h2_train": h2_train,
        "phi_train": phi_train,
        "mu_cal": mu_cal,
        "h2_cal": h2_cal,
        "phi_cal": phi_cal,
        "mu_test": mu_test,
        "h2_test": h2_test,
        "phi_test": phi_test,
    }
    return predictions, float(elapsed)


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
    cfg: RepresentationDimensionConfig,
) -> np.ndarray:
    h_augmented = add_bias_column(phi_train)
    weights = 1.0 / np.maximum(
        h2_train.reshape(-1).astype(np.float64),
        cfg.variance_floor,
    )

    precision = prior_precision * np.eye(
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
    cfg: RepresentationDimensionConfig,
) -> np.ndarray:
    precision = build_weighted_precision(
        phi_train=phi_train,
        h2_train=h2_train,
        prior_precision=prior_precision,
        cfg=cfg,
    )
    return invert_positive_semidefinite(precision)


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
    feature_dim: int,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    cfg: RepresentationDimensionConfig,
) -> Tuple[float, List[Dict[str, float]], float]:
    # The inner split is deterministic and identical across feature dimensions.
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
        covariance = compute_weighted_laplace_covariance(
            phi_train=predictions["phi_train"][fit_idx],
            h2_train=predictions["h2_train"][fit_idx],
            prior_precision=float(prior_precision),
            cfg=cfg,
        )
        epistemic_validation = quadratic_form(phi_validation, covariance)
        total_variance_validation = np.maximum(
            h2_validation + epistemic_validation,
            cfg.variance_floor,
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
                "feature_dim": int(feature_dim),
                "prior_precision": float(prior_precision),
                "validation_nll": validation_nll,
                "inner_fit_size": int(len(fit_idx)),
                "inner_validation_size": int(len(validation_idx)),
            }
        )

        if validation_nll < best_nll:
            best_nll = validation_nll
            best_lambda = float(prior_precision)

    selection_time_sec = time.perf_counter() - start_time

    for row in trace_rows:
        row["selected_prior_precision"] = best_lambda
        row["is_selected"] = bool(
            np.isclose(row["prior_precision"], best_lambda)
        )
        row["nll_excess_over_best"] = row["validation_nll"] - best_nll

    return best_lambda, trace_rows, float(selection_time_sec)


def timed_final_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    cfg: RepresentationDimensionConfig,
) -> Tuple[np.ndarray, float, float]:
    start_time = time.perf_counter()
    precision = build_weighted_precision(
        phi_train=phi_train,
        h2_train=h2_train,
        prior_precision=prior_precision,
        cfg=cfg,
    )
    covariance = invert_positive_semidefinite(precision)
    elapsed_ms = 1000.0 * (time.perf_counter() - start_time)

    # Condition number is diagnostic only and is not included in the timed
    # covariance-construction measurement.
    condition_number = float(np.linalg.cond(precision))
    return covariance, float(elapsed_ms), condition_number


def benchmark_quadratic_form(
    phi_test: np.ndarray,
    covariance: np.ndarray,
    cfg: RepresentationDimensionConfig,
) -> Tuple[np.ndarray, float, float]:
    # Warm-up.
    epistemic_test = quadratic_form(phi_test, covariance)

    repeats = max(int(cfg.timing_repeats), 1)
    elapsed_values = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        start_time = time.perf_counter()
        epistemic_test = quadratic_form(phi_test, covariance)
        elapsed_values[repeat] = time.perf_counter() - start_time

    median_time_sec = float(np.median(elapsed_values))
    batch_time_ms = 1000.0 * median_time_sec
    per_sample_us = 1e6 * median_time_sec / max(len(phi_test), 1)
    return epistemic_test, float(batch_time_ms), float(per_sample_us)


# ============================================================
# Independent learned-feature support quintiles
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
    estimator = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )
    estimator.fit(train_features)
    distances, _ = estimator.kneighbors(test_features, return_distance=True)
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
    cfg: RepresentationDimensionConfig,
) -> np.ndarray:
    phi_train_standardized, phi_test_standardized = standardize_feature_space(
        phi_train,
        phi_test,
    )
    support_distance = knn_support_distance(
        phi_train_standardized,
        phi_test_standardized,
        cfg.support_k,
    )
    # Q1 = strongest learned-feature support; Q5 = weakest support.
    return rank_based_groups(support_distance, cfg.support_quintiles)


# ============================================================
# LACP and CLAPS intervals
# ============================================================


def interval_from_scale(
    mu_cal: np.ndarray,
    scale_cal: np.ndarray,
    mu_test: np.ndarray,
    scale_test: np.ndarray,
    data: Mapping[str, np.ndarray],
    cfg: RepresentationDimensionConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scale_cal = np.maximum(scale_cal, 1e-8)
    scale_test = np.maximum(scale_test, 1e-8)

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
    feature_dim: int,
    method: str,
    selected_prior_precision: float,
    data: Mapping[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float,
    support_quintiles: np.ndarray,
    epistemic_fraction: float,
    q5_epistemic_fraction: float,
    computation: Mapping[str, float],
    cfg: RepresentationDimensionConfig,
) -> Dict[str, float]:
    y = np.asarray(data["y_test"]).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    scores = interval_score(y, lower, upper, cfg.alpha)
    y_scale = max(float(data["y_scale"]), 1e-12)

    q5_mask = support_quintiles == cfg.support_quintiles
    if not np.any(q5_mask):
        raise RuntimeError("The weakest learned-support quintile is empty.")

    row = {
        "dataset": dataset_name,
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "augmented_feature_dim": int(feature_dim + 1),
        "method": method,
        "selected_prior_precision": float(selected_prior_precision),
        "n_train": int(data["n_train_actual"]),
        "n_cal": int(data["n_cal_actual"]),
        "n_test": int(data["n_test_actual"]),
        "marginal_coverage": float(np.mean(covered)),
        "normalized_width": float(np.mean(width) / y_scale),
        "normalized_interval_score": float(np.mean(scores) / y_scale),
        "q5_coverage": float(np.mean(covered[q5_mask])),
        "q5_normalized_width": float(np.mean(width[q5_mask]) / y_scale),
        "q5_normalized_interval_score": float(
            np.mean(scores[q5_mask]) / y_scale
        ),
        "qhat": float(qhat),
        "epistemic_fraction": float(epistemic_fraction),
        "q5_epistemic_fraction": float(q5_epistemic_fraction),
    }
    row.update({key: float(value) for key, value in computation.items()})
    return row


def run_lacp_from_predictions(
    dataset_name: str,
    seed: int,
    feature_dim: int,
    selected_prior_precision: float,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    support_quintiles: np.ndarray,
    computation: Mapping[str, float],
    cfg: RepresentationDimensionConfig,
) -> Dict[str, float]:
    lower, upper, qhat = interval_from_scale(
        mu_cal=predictions["mu_cal"],
        scale_cal=np.sqrt(
            np.maximum(predictions["h2_cal"], cfg.variance_floor)
        ),
        mu_test=predictions["mu_test"],
        scale_test=np.sqrt(
            np.maximum(predictions["h2_test"], cfg.variance_floor)
        ),
        data=data,
        cfg=cfg,
    )

    return compute_interval_metrics(
        dataset_name=dataset_name,
        seed=seed,
        feature_dim=feature_dim,
        method="LACP",
        selected_prior_precision=selected_prior_precision,
        data=data,
        lower=lower,
        upper=upper,
        qhat=qhat,
        support_quintiles=support_quintiles,
        epistemic_fraction=np.nan,
        q5_epistemic_fraction=np.nan,
        computation=computation,
        cfg=cfg,
    )


def run_claps_from_predictions(
    dataset_name: str,
    seed: int,
    feature_dim: int,
    selected_prior_precision: float,
    data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    support_quintiles: np.ndarray,
    base_computation: Mapping[str, float],
    cfg: RepresentationDimensionConfig,
) -> Dict[str, float]:
    covariance, covariance_time_ms, condition_number = timed_final_covariance(
        phi_train=predictions["phi_train"],
        h2_train=predictions["h2_train"],
        prior_precision=selected_prior_precision,
        cfg=cfg,
    )

    calibration_start = time.perf_counter()
    epistemic_calibration = quadratic_form(
        predictions["phi_cal"],
        covariance,
    )
    calibration_quadratic_time_ms = 1000.0 * (
        time.perf_counter() - calibration_start
    )

    epistemic_test, test_batch_time_ms, test_per_sample_us = (
        benchmark_quadratic_form(
            predictions["phi_test"],
            covariance,
            cfg,
        )
    )

    total_variance_calibration = np.maximum(
        predictions["h2_cal"] + epistemic_calibration,
        cfg.variance_floor,
    )
    total_variance_test = np.maximum(
        predictions["h2_test"] + epistemic_test,
        cfg.variance_floor,
    )

    lower, upper, qhat = interval_from_scale(
        mu_cal=predictions["mu_cal"],
        scale_cal=np.sqrt(total_variance_calibration),
        mu_test=predictions["mu_test"],
        scale_test=np.sqrt(total_variance_test),
        data=data,
        cfg=cfg,
    )

    pointwise_fraction = epistemic_test / np.maximum(
        total_variance_test,
        cfg.variance_floor,
    )
    q5_mask = support_quintiles == cfg.support_quintiles

    computation = dict(base_computation)
    computation.update(
        {
            "covariance_construction_time_ms": covariance_time_ms,
            "calibration_quadratic_time_ms": float(
                calibration_quadratic_time_ms
            ),
            "test_quadratic_batch_time_ms": test_batch_time_ms,
            "test_quadratic_us_per_sample": test_per_sample_us,
            "covariance_memory_mb": float(covariance.nbytes / (1024.0**2)),
            "precision_condition_number": condition_number,
        }
    )

    return compute_interval_metrics(
        dataset_name=dataset_name,
        seed=seed,
        feature_dim=feature_dim,
        method="CLAPS",
        selected_prior_precision=selected_prior_precision,
        data=data,
        lower=lower,
        upper=upper,
        qhat=qhat,
        support_quintiles=support_quintiles,
        epistemic_fraction=float(np.mean(pointwise_fraction)),
        q5_epistemic_fraction=float(np.mean(pointwise_fraction[q5_mask])),
        computation=computation,
        cfg=cfg,
    )


# ============================================================
# One dimension within one dataset-seed run
# ============================================================


def run_single_dimension(
    dataset_name: str,
    data: Mapping[str, np.ndarray],
    seed: int,
    feature_dim: int,
    cfg: RepresentationDimensionConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    model, training_diagnostics = train_hetero_model(
        x=data["x_train"],
        y=data["y_train_std"],
        feature_dim=feature_dim,
        seed=seed,
        cfg=cfg,
    )

    predictions, feature_extraction_time_sec = predict_all_splits(
        model,
        data,
    )

    selected_lambda, selection_trace, selection_time_sec = (
        select_prior_precision_training_only(
            dataset_name=dataset_name,
            seed=seed,
            feature_dim=feature_dim,
            data=data,
            predictions=predictions,
            cfg=cfg,
        )
    )

    support_quintiles = learned_feature_support_quintiles(
        phi_train=predictions["phi_train"],
        phi_test=predictions["phi_test"],
        cfg=cfg,
    )

    base_computation = {
        "training_time_sec": training_diagnostics["training_time_sec"],
        "completed_epochs": training_diagnostics["completed_epochs"],
        "best_validation_loss": training_diagnostics[
            "best_validation_loss"
        ],
        "parameter_count": training_diagnostics["parameter_count"],
        "feature_extraction_time_sec": feature_extraction_time_sec,
        "lambda_selection_time_sec": selection_time_sec,
        "covariance_construction_time_ms": np.nan,
        "calibration_quadratic_time_ms": np.nan,
        "test_quadratic_batch_time_ms": np.nan,
        "test_quadratic_us_per_sample": np.nan,
        "covariance_memory_mb": np.nan,
        "precision_condition_number": np.nan,
    }

    rows = [
        run_lacp_from_predictions(
            dataset_name=dataset_name,
            seed=seed,
            feature_dim=feature_dim,
            selected_prior_precision=selected_lambda,
            data=data,
            predictions=predictions,
            support_quintiles=support_quintiles,
            computation=base_computation,
            cfg=cfg,
        ),
        run_claps_from_predictions(
            dataset_name=dataset_name,
            seed=seed,
            feature_dim=feature_dim,
            selected_prior_precision=selected_lambda,
            data=data,
            predictions=predictions,
            support_quintiles=support_quintiles,
            base_computation=base_computation,
            cfg=cfg,
        ),
    ]

    del model, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, selection_trace


# ============================================================
# Main experiment loop
# ============================================================


def run_experiment(
    cfg: RepresentationDimensionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = load_all_datasets(cfg)
    result_rows: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, float]] = []

    total_jobs = (
        len(datasets) * cfg.num_seeds * len(cfg.feature_dim_grid)
    )
    completed = 0
    start_time = time.time()

    for dataset_name, (x, y) in datasets.items():
        for seed in range(cfg.num_seeds):
            # The same outer split and standardization are reused for all
            # representation dimensions.
            data = make_random_split(x, y, seed, cfg)

            for feature_dim in cfg.feature_dim_grid:
                completed += 1
                if cfg.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"[{completed}/{total_jobs}] "
                        f"Dataset={dataset_name} | Seed={seed} | "
                        f"Feature dim={feature_dim} | "
                        f"Elapsed={elapsed:.1f}s"
                    )

                run_rows, trace_rows = run_single_dimension(
                    dataset_name=dataset_name,
                    data=data,
                    seed=seed,
                    feature_dim=int(feature_dim),
                    cfg=cfg,
                )
                result_rows.extend(run_rows)
                selection_rows.extend(trace_rows)

            del data
            gc.collect()

    return pd.DataFrame(result_rows), pd.DataFrame(selection_rows)


# ============================================================
# LACP-relative effects
# ============================================================


def add_lacp_relative_effects(results: pd.DataFrame) -> pd.DataFrame:
    lacp = (
        results[results["method"] == "LACP"]
        [
            [
                "dataset",
                "seed",
                "feature_dim",
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
        on=["dataset", "seed", "feature_dim"],
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
    ("epistemic_fraction", "Epistemic Fraction", 100.0, "%", 2),
    (
        "q5_epistemic_fraction",
        "Weak-Q5 Epistemic Fraction",
        100.0,
        "%",
        2,
    ),
)


def summarize_performance_by_dimension(
    results: pd.DataFrame,
    cfg: RepresentationDimensionConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    rows: List[Dict[str, str]] = []

    for feature_dim in cfg.feature_dim_grid:
        subset = claps[claps["feature_dim"] == int(feature_dim)]
        if subset.empty:
            continue

        row: Dict[str, str] = {
            "Feature Dim": str(int(feature_dim)),
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


def summarize_performance_by_dataset_dimension(
    results: pd.DataFrame,
    cfg: RepresentationDimensionConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()
    rows: List[Dict[str, str]] = []

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
    )

    for dataset_name in cfg.dataset_names:
        for feature_dim in cfg.feature_dim_grid:
            subset = claps[
                (claps["dataset"] == dataset_name)
                & (claps["feature_dim"] == int(feature_dim))
            ]
            if subset.empty:
                continue

            row: Dict[str, str] = {
                "Dataset": dataset_name,
                "Feature Dim": str(int(feature_dim)),
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


def summarize_computation_by_dimension(
    results: pd.DataFrame,
    cfg: RepresentationDimensionConfig,
) -> pd.DataFrame:
    claps = results[results["method"] == "CLAPS"].copy()

    metric_specs = (
        ("parameter_count", "Parameters", 1.0, "", 0),
        ("completed_epochs", "Epochs", 1.0, "", 1),
        ("training_time_sec", "Training Time", 1.0, " s", 2),
        (
            "feature_extraction_time_sec",
            "Feature Extraction Time",
            1.0,
            " s",
            3,
        ),
        (
            "lambda_selection_time_sec",
            "Lambda Selection Time",
            1.0,
            " s",
            3,
        ),
        (
            "covariance_construction_time_ms",
            "Covariance Construction",
            1.0,
            " ms",
            3,
        ),
        (
            "test_quadratic_batch_time_ms",
            "Test Quadratic Batch",
            1.0,
            " ms",
            3,
        ),
        (
            "test_quadratic_us_per_sample",
            "Test Quadratic per Sample",
            1.0,
            " us",
            3,
        ),
        (
            "covariance_memory_mb",
            "Covariance Memory",
            1.0,
            " MB",
            4,
        ),
    )

    rows: List[Dict[str, str]] = []
    for feature_dim in cfg.feature_dim_grid:
        subset = claps[claps["feature_dim"] == int(feature_dim)]
        if subset.empty:
            continue

        row: Dict[str, str] = {
            "Feature Dim": str(int(feature_dim)),
            "Augmented Dim": str(int(feature_dim) + 1),
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


def build_lambda_selection_frequency(
    selection_trace: pd.DataFrame,
    cfg: RepresentationDimensionConfig,
) -> pd.DataFrame:
    selected = (
        selection_trace[selection_trace["is_selected"]]
        [["dataset", "seed", "feature_dim", "prior_precision"]]
        .drop_duplicates()
    )

    rows = []
    for feature_dim in cfg.feature_dim_grid:
        dimension_selected = selected[
            selected["feature_dim"] == int(feature_dim)
        ]
        denominator = len(dimension_selected)
        for prior_precision in cfg.lambda_grid:
            count = int(
                np.sum(
                    np.isclose(
                        dimension_selected["prior_precision"],
                        prior_precision,
                    )
                )
            )
            rows.append(
                {
                    "Feature Dim": int(feature_dim),
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
# Hierarchical paired bootstrap by dimension
# ============================================================


def hierarchical_bootstrap_mean(
    data: pd.DataFrame,
    metric: str,
    cfg: RepresentationDimensionConfig,
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

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )
    return observed, float(ci_low), float(ci_high)


def build_dimension_bootstrap_table(
    results: pd.DataFrame,
    cfg: RepresentationDimensionConfig,
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
    for dimension_index, feature_dim in enumerate(cfg.feature_dim_grid):
        subset = claps[claps["feature_dim"] == int(feature_dim)]
        if subset.empty:
            continue

        for metric_index, (metric, label, multiplier, suffix) in enumerate(
            metric_specs
        ):
            mean_value, ci_low, ci_high = hierarchical_bootstrap_mean(
                subset,
                metric,
                cfg,
                seed_offset=1000 * dimension_index + metric_index,
            )
            rows.append(
                {
                    "Feature Dim": int(feature_dim),
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
# Saving
# ============================================================


def save_outputs(
    output_frames: Mapping[str, pd.DataFrame],
    cfg: RepresentationDimensionConfig,
) -> None:
    if not cfg.save_csv:
        return

    for suffix, frame in output_frames.items():
        path = f"{cfg.output_prefix}_{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")


# ============================================================
# Main entry point
# ============================================================


def main(
    cfg: RepresentationDimensionConfig = CFG,
) -> Dict[str, pd.DataFrame]:
    prepare_packages()
    results_df, selection_trace_df = run_experiment(cfg)
    results_df = add_lacp_relative_effects(results_df)

    dimension_summary_df = summarize_performance_by_dimension(
        results_df,
        cfg,
    )
    dataset_dimension_summary_df = (
        summarize_performance_by_dataset_dimension(results_df, cfg)
    )
    computation_summary_df = summarize_computation_by_dimension(
        results_df,
        cfg,
    )
    lambda_selection_frequency_df = build_lambda_selection_frequency(
        selection_trace_df,
        cfg,
    )
    dimension_bootstrap_df = build_dimension_bootstrap_table(
        results_df,
        cfg,
    )

    output_frames = {
        "raw_results": results_df,
        "selection_trace": selection_trace_df,
        "dimension_summary": dimension_summary_df,
        "dataset_dimension_summary": dataset_dimension_summary_df,
        "computation_summary": computation_summary_df,
        "lambda_selection_frequency": lambda_selection_frequency_df,
        "dimension_bootstrap": dimension_bootstrap_df,
    }
    save_outputs(output_frames, cfg)

    print("\nFinished representation-dimension sensitivity analysis.")

    print("\nPerformance sensitivity: mean ± standard error")
    display(dimension_summary_df)

    print("\nComputational cost by representation dimension")
    display(computation_summary_df)

    print("\nTraining-only prior-precision selection frequency")
    display(lambda_selection_frequency_df)

    print("\nHierarchical paired bootstrap: CLAPS versus matched LACP")
    display(dimension_bootstrap_df)

    print("\nDataset-level sensitivity")
    for dataset_name in cfg.dataset_names:
        print(f"\nDataset: {dataset_name}")
        display(
            dataset_dimension_summary_df[
                dataset_dimension_summary_df["Dataset"] == dataset_name
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
    dimension_summary_df = OUTPUTS["dimension_summary"]
    dataset_dimension_summary_df = OUTPUTS[
        "dataset_dimension_summary"
    ]
    computation_summary_df = OUTPUTS["computation_summary"]
    lambda_selection_frequency_df = OUTPUTS[
        "lambda_selection_frequency"
    ]
    dimension_bootstrap_df = OUTPUTS["dimension_bootstrap"]


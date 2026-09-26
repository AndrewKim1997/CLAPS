
# -*- coding: utf-8 -*-
"""
Appendix: Prior-Precision Selection and Sensitivity for CLAPS
==============================================================

Standalone Google Colab / Python implementation aligned with the revised
real-data experiment.

Protocol
--------
1. Use the same eight tabular datasets and 60/20/20 train/calibration/test
   protocol as the revised real-data benchmark.
2. Fit one heteroscedastic neural regressor per dataset and seed.
3. Select the CLAPS prior precision only from a training-only inner split by
   predictive negative log-likelihood. Calibration and test responses are
   never used for selection.
4. Recompute the last-layer covariance from the full training split for every
   fixed prior precision and for the selected prior precision.
5. Compare every CLAPS configuration with the matched LACP interval obtained
   from the same fitted regressor.
6. Report aggregate and weakest learned-feature-support-quintile effects.

Created outputs
---------------
results_df
    Per-dataset, per-seed LACP and CLAPS results.
selection_trace_df
    Training-only validation NLL for every candidate prior precision.
fixed_grid_summary_df
    Fixed-grid sensitivity summary, mean ± standard error.
selected_summary_df
    Training-only selected-prior summary, mean ± standard error.
selection_frequency_df
    Overall selection frequencies.
dataset_selection_frequency_df
    Dataset-specific selection frequencies.
selection_nll_summary_df
    Mean validation-NLL excess over the best candidate.
selected_bootstrap_df
    Hierarchical paired bootstrap comparison of selected CLAPS versus LACP.

The code intentionally trains only the shared heteroscedastic model. It does
not rerun unrelated baselines.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
class PriorPrecisionConfig:
    # Same benchmark datasets as revised Experiment 4.
    dataset_names: Tuple[str, ...] = (
        "Concrete",
        "Energy",
        "Yacht",
        "Airfoil",
        "WineRed",
        "Naval",
        "Bike",
        "California",
    )

    # Final paper protocol.
    num_seeds: int = 20
    train_fraction: float = 0.60
    cal_fraction: float = 0.20
    test_fraction: float = 0.20
    max_samples_per_dataset: int = 5000

    # Nominal coverage.
    alpha: float = 0.10

    # Shared heteroscedastic neural regressor.
    hidden_dim: int = 64
    feature_dim: int = 64
    max_epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5

    # The selection grid exactly matches the revised main experiment.
    selection_lambda_grid: Tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    )

    # The sensitivity grid extends the main grid at both ends.
    sensitivity_lambda_grid: Tuple[float, ...] = (
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    )

    # Last-layer Laplace.
    variance_floor: float = 1e-4
    jitter: float = 1e-6

    # Independent support proxy, matching revised Experiment 4.
    support_k: int = 20
    support_quintiles: int = 5

    # Hierarchical paired bootstrap.
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 20260801

    # Display and saving.
    verbose: bool = True
    summary_digits: int = 4
    save_csv: bool = True
    output_prefix: str = "prior_precision"
    display_raw_results: bool = False


CFG = PriorPrecisionConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("Appendix experiment: prior-precision selection and sensitivity")
print(f"Datasets: {CFG.dataset_names}")
print(f"Selection grid: {CFG.selection_lambda_grid}")
print(f"Sensitivity grid: {CFG.sensitivity_lambda_grid}")


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
        raise ValueError("At least two training observations are required.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * validation_fraction)))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]
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


def lambda_label(value: float) -> str:
    return f"{float(value):g}"


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
    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
    y_series = pd.to_numeric(y_series, errors="coerce")

    df = x_df.copy()
    df["_target"] = y_series.to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    y = df["_target"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x = df.drop(columns=["_target"]).to_numpy(dtype=np.float32)
    return x, y


def load_concrete() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "concrete/compressive/Concrete_Data.xls"
    )
    df = pd.read_excel(url, engine="xlrd")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_energy_efficiency() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00242/ENB2012_data.xlsx"
    )
    df = pd.read_excel(url, engine="openpyxl")
    return clean_xy(df.iloc[:, :8], df.iloc[:, 8])


def load_yacht_hydrodynamics() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00243/yacht_hydrodynamics.data"
    )
    df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_airfoil_self_noise() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00291/airfoil_self_noise.dat"
    )
    df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_wine_quality_red() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    df = pd.read_csv(url, sep=";")
    return clean_xy(df.drop(columns=["quality"]), df["quality"])


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

    df = pd.read_csv(
        archive.open(data_filename),
        sep=r"\s+",
        header=None,
        engine="python",
    ).dropna(axis=1, how="all")

    # Columns 0--15 are features; column 17 is GT turbine decay coefficient.
    return clean_xy(df.iloc[:, :16], df.iloc[:, 17])


def load_bike_sharing() -> Tuple[np.ndarray, np.ndarray]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00275/Bike-Sharing-Dataset.zip"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(BytesIO(response.content))
    df = pd.read_csv(archive.open("hour.csv"))

    target = df["cnt"]
    drop_cols = [
        col
        for col in ["cnt", "casual", "registered", "instant", "dteday"]
        if col in df.columns
    ]
    return clean_xy(df.drop(columns=drop_cols), target)


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing(as_frame=True)
    return clean_xy(data.data, data.target)


def load_all_datasets(
    cfg: PriorPrecisionConfig,
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
    cfg: PriorPrecisionConfig,
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
    data["n_train_actual"] = len(train_idx)
    data["n_cal_actual"] = len(cal_idx)
    data["n_test_actual"] = len(test_idx)
    return data


# ============================================================
# Heteroscedastic neural regressor
# ============================================================


class HeteroMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        variance_floor: float,
    ) -> None:
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
    seed: int,
    cfg: PriorPrecisionConfig,
) -> HeteroMLP:
    # A dedicated seed makes this model independent of any unrelated baseline
    # training order in another notebook cell.
    set_seed(100_000 + seed)

    model = HeteroMLP(
        input_dim=x.shape[1],
        hidden_dim=cfg.hidden_dim,
        feature_dim=cfg.feature_dim,
        variance_floor=cfg.variance_floor,
    ).to(DEVICE)

    x_t = to_tensor(x)
    y_t = to_tensor(y)
    fit_idx, val_idx = deterministic_split(
        len(x),
        cfg.validation_fraction,
        seed=200_000 + seed,
    )
    fit_idx_t = torch.tensor(fit_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()

        mu, h2, _ = model(x_t[fit_idx_t])
        residual2 = (y_t[fit_idx_t] - mu) ** 2
        loss = 0.5 * torch.mean(residual2 / h2 + torch.log(h2))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mu, val_h2, _ = model(x_t[val_idx_t])
            val_residual2 = (y_t[val_idx_t] - val_mu) ** 2
            val_loss = 0.5 * torch.mean(
                val_residual2 / val_h2 + torch.log(val_h2)
            ).item()

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


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


# ============================================================
# Last-layer covariance and training-only selection
# ============================================================


def add_bias_column(phi: np.ndarray) -> np.ndarray:
    ones = np.ones((phi.shape[0], 1), dtype=np.float64)
    return np.concatenate([phi.astype(np.float64), ones], axis=1)


def invert_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def compute_weighted_laplace_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    cfg: PriorPrecisionConfig,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train)
    weights = 1.0 / np.maximum(
        h2_train.reshape(-1).astype(np.float64),
        cfg.variance_floor,
    )

    precision = float(prior_precision) * np.eye(
        h_aug.shape[1],
        dtype=np.float64,
    )
    precision += h_aug.T @ (h_aug * weights[:, None])
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)
    return invert_positive_semidefinite(precision)


def quadratic_form(
    phi: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    h_aug = add_bias_column(phi)
    value = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(value, 0.0).astype(np.float32)


def select_prior_precision_training_only(
    dataset_name: str,
    seed: int,
    data: Mapping[str, np.ndarray],
    mu_train: np.ndarray,
    h2_train: np.ndarray,
    phi_train: np.ndarray,
    cfg: PriorPrecisionConfig,
) -> Tuple[float, List[Dict[str, float]]]:
    """Select lambda without using calibration or test labels.

    The fitted representation and variance head are held fixed. The training
    split is divided into an inner covariance-fit subset and an inner
    validation subset. Candidate lambdas are evaluated by predictive Gaussian
    NLL on the inner validation subset. After selection, the final covariance
    is recomputed from the full training split elsewhere in the code.
    """

    fit_idx, val_idx = deterministic_split(
        len(data["x_train"]),
        cfg.validation_fraction,
        seed=300_000 + seed,
    )

    y_val = data["y_train_std"][val_idx]
    mu_val = mu_train[val_idx]
    h2_val = h2_train[val_idx]
    phi_val = phi_train[val_idx]

    traces: List[Dict[str, float]] = []
    best_lambda = float(cfg.selection_lambda_grid[0])
    best_nll = float("inf")

    for lam in cfg.selection_lambda_grid:
        covariance = compute_weighted_laplace_covariance(
            phi_train=phi_train[fit_idx],
            h2_train=h2_train[fit_idx],
            prior_precision=float(lam),
            cfg=cfg,
        )
        epi_val = quadratic_form(phi_val, covariance)
        total_var_val = np.maximum(h2_val + epi_val, cfg.variance_floor)
        residual2 = (y_val - mu_val) ** 2
        validation_nll = float(
            0.5 * np.mean(
                residual2 / total_var_val + np.log(total_var_val)
            )
        )

        traces.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "prior_precision": float(lam),
                "validation_nll": validation_nll,
                "inner_fit_size": int(len(fit_idx)),
                "inner_validation_size": int(len(val_idx)),
            }
        )

        if validation_nll < best_nll:
            best_nll = validation_nll
            best_lambda = float(lam)

    for row in traces:
        row["selected_prior_precision"] = best_lambda
        row["is_selected"] = bool(
            np.isclose(row["prior_precision"], best_lambda)
        )
        row["nll_excess_over_best"] = row["validation_nll"] - best_nll

    return best_lambda, traces


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
    cfg: PriorPrecisionConfig,
) -> np.ndarray:
    phi_train_std, phi_test_std = standardize_feature_space(
        phi_train,
        phi_test,
    )
    distance = knn_support_distance(
        phi_train_std,
        phi_test_std,
        cfg.support_k,
    )
    # Q1 = strongest support; Q5 = weakest support.
    return rank_based_groups(distance, cfg.support_quintiles)


# ============================================================
# LACP and CLAPS interval construction
# ============================================================


def interval_from_scale(
    mu_cal: np.ndarray,
    scale_cal: np.ndarray,
    mu_test: np.ndarray,
    scale_test: np.ndarray,
    data: Mapping[str, np.ndarray],
    cfg: PriorPrecisionConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scale_cal = np.maximum(scale_cal, 1e-8)
    scale_test = np.maximum(scale_test, 1e-8)

    scores = np.abs(data["y_cal_std"] - mu_cal) / scale_cal
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_std = mu_test - qhat * scale_test
    upper_std = mu_test + qhat * scale_test

    lower = unstandardize_y(
        lower_std,
        float(data["y_mean"]),
        float(data["y_scale"]),
    )
    upper = unstandardize_y(
        upper_std,
        float(data["y_mean"]),
        float(data["y_scale"]),
    )
    return lower, upper, float(qhat)


def compute_interval_metrics(
    dataset_name: str,
    seed: int,
    method: str,
    configuration: str,
    selection_type: str,
    prior_precision: float,
    selected_prior_precision: float,
    data: Mapping[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float,
    support_quintiles: np.ndarray,
    epistemic_fraction: float,
    q5_epistemic_fraction: float,
    cfg: PriorPrecisionConfig,
) -> Dict[str, float]:
    y = np.asarray(data["y_test"]).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = interval_score(y, lower, upper, cfg.alpha)
    y_scale = max(float(data["y_scale"]), 1e-12)

    q5_mask = support_quintiles == cfg.support_quintiles
    if not np.any(q5_mask):
        raise RuntimeError("The weakest support quintile is empty.")

    return {
        "dataset": dataset_name,
        "seed": int(seed),
        "method": method,
        "configuration": configuration,
        "selection_type": selection_type,
        "prior_precision": float(prior_precision),
        "selected_prior_precision": float(selected_prior_precision),
        "n_train": int(data["n_train_actual"]),
        "n_cal": int(data["n_cal_actual"]),
        "n_test": int(data["n_test_actual"]),
        "marginal_coverage": float(np.mean(covered)),
        "normalized_width": float(np.mean(width) / y_scale),
        "normalized_interval_score": float(np.mean(score) / y_scale),
        "q5_coverage": float(np.mean(covered[q5_mask])),
        "q5_normalized_width": float(np.mean(width[q5_mask]) / y_scale),
        "q5_normalized_interval_score": float(
            np.mean(score[q5_mask]) / y_scale
        ),
        "qhat": float(qhat),
        "epistemic_fraction": float(epistemic_fraction),
        "q5_epistemic_fraction": float(q5_epistemic_fraction),
    }


def run_lacp_from_predictions(
    dataset_name: str,
    seed: int,
    data: Mapping[str, np.ndarray],
    mu_cal: np.ndarray,
    h2_cal: np.ndarray,
    mu_test: np.ndarray,
    h2_test: np.ndarray,
    support_quintiles: np.ndarray,
    selected_prior_precision: float,
    cfg: PriorPrecisionConfig,
) -> Dict[str, float]:
    lower, upper, qhat = interval_from_scale(
        mu_cal=mu_cal,
        scale_cal=np.sqrt(np.maximum(h2_cal, cfg.variance_floor)),
        mu_test=mu_test,
        scale_test=np.sqrt(np.maximum(h2_test, cfg.variance_floor)),
        data=data,
        cfg=cfg,
    )

    return compute_interval_metrics(
        dataset_name=dataset_name,
        seed=seed,
        method="LACP",
        configuration="LACP",
        selection_type="Baseline",
        prior_precision=np.nan,
        selected_prior_precision=selected_prior_precision,
        data=data,
        lower=lower,
        upper=upper,
        qhat=qhat,
        support_quintiles=support_quintiles,
        epistemic_fraction=np.nan,
        q5_epistemic_fraction=np.nan,
        cfg=cfg,
    )


def run_claps_from_predictions(
    dataset_name: str,
    seed: int,
    data: Mapping[str, np.ndarray],
    h2_train: np.ndarray,
    phi_train: np.ndarray,
    mu_cal: np.ndarray,
    h2_cal: np.ndarray,
    phi_cal: np.ndarray,
    mu_test: np.ndarray,
    h2_test: np.ndarray,
    phi_test: np.ndarray,
    support_quintiles: np.ndarray,
    prior_precision: float,
    selected_prior_precision: float,
    configuration: str,
    selection_type: str,
    cfg: PriorPrecisionConfig,
) -> Dict[str, float]:
    covariance = compute_weighted_laplace_covariance(
        phi_train=phi_train,
        h2_train=h2_train,
        prior_precision=prior_precision,
        cfg=cfg,
    )
    epi_cal = quadratic_form(phi_cal, covariance)
    epi_test = quadratic_form(phi_test, covariance)

    total_var_cal = np.maximum(h2_cal + epi_cal, cfg.variance_floor)
    total_var_test = np.maximum(h2_test + epi_test, cfg.variance_floor)

    lower, upper, qhat = interval_from_scale(
        mu_cal=mu_cal,
        scale_cal=np.sqrt(total_var_cal),
        mu_test=mu_test,
        scale_test=np.sqrt(total_var_test),
        data=data,
        cfg=cfg,
    )

    pointwise_fraction = epi_test / np.maximum(
        total_var_test,
        cfg.variance_floor,
    )
    q5_mask = support_quintiles == cfg.support_quintiles

    return compute_interval_metrics(
        dataset_name=dataset_name,
        seed=seed,
        method="CLAPS",
        configuration=configuration,
        selection_type=selection_type,
        prior_precision=prior_precision,
        selected_prior_precision=selected_prior_precision,
        data=data,
        lower=lower,
        upper=upper,
        qhat=qhat,
        support_quintiles=support_quintiles,
        epistemic_fraction=float(np.mean(pointwise_fraction)),
        q5_epistemic_fraction=float(np.mean(pointwise_fraction[q5_mask])),
        cfg=cfg,
    )


# ============================================================
# One dataset-seed run
# ============================================================


def run_single_dataset_seed(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: PriorPrecisionConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    data = make_random_split(x, y, seed, cfg)
    model = train_hetero_model(
        x=data["x_train"],
        y=data["y_train_std"],
        seed=seed,
        cfg=cfg,
    )

    mu_train, h2_train, phi_train = predict_hetero(
        model,
        data["x_train"],
    )
    mu_cal, h2_cal, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, phi_test = predict_hetero(model, data["x_test"])

    selected_lambda, selection_trace = select_prior_precision_training_only(
        dataset_name=dataset_name,
        seed=seed,
        data=data,
        mu_train=mu_train,
        h2_train=h2_train,
        phi_train=phi_train,
        cfg=cfg,
    )

    support_quintiles = learned_feature_support_quintiles(
        phi_train=phi_train,
        phi_test=phi_test,
        cfg=cfg,
    )

    rows: List[Dict[str, float]] = []
    rows.append(
        run_lacp_from_predictions(
            dataset_name=dataset_name,
            seed=seed,
            data=data,
            mu_cal=mu_cal,
            h2_cal=h2_cal,
            mu_test=mu_test,
            h2_test=h2_test,
            support_quintiles=support_quintiles,
            selected_prior_precision=selected_lambda,
            cfg=cfg,
        )
    )

    for lam in cfg.sensitivity_lambda_grid:
        rows.append(
            run_claps_from_predictions(
                dataset_name=dataset_name,
                seed=seed,
                data=data,
                h2_train=h2_train,
                phi_train=phi_train,
                mu_cal=mu_cal,
                h2_cal=h2_cal,
                phi_cal=phi_cal,
                mu_test=mu_test,
                h2_test=h2_test,
                phi_test=phi_test,
                support_quintiles=support_quintiles,
                prior_precision=float(lam),
                selected_prior_precision=selected_lambda,
                configuration=f"lambda={lambda_label(lam)}",
                selection_type="Fixed",
                cfg=cfg,
            )
        )

    rows.append(
        run_claps_from_predictions(
            dataset_name=dataset_name,
            seed=seed,
            data=data,
            h2_train=h2_train,
            phi_train=phi_train,
            mu_cal=mu_cal,
            h2_cal=h2_cal,
            phi_cal=phi_cal,
            mu_test=mu_test,
            h2_test=h2_test,
            phi_test=phi_test,
            support_quintiles=support_quintiles,
            prior_precision=selected_lambda,
            selected_prior_precision=selected_lambda,
            configuration="Selected",
            selection_type="Selected",
            cfg=cfg,
        )
    )

    del model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, selection_trace


# ============================================================
# Main experiment loop
# ============================================================


def run_experiment(
    cfg: PriorPrecisionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = load_all_datasets(cfg)
    result_rows: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, float]] = []

    total_jobs = len(datasets) * cfg.num_seeds
    completed = 0
    start_time = time.time()

    for dataset_name, (x, y) in datasets.items():
        for seed in range(cfg.num_seeds):
            completed += 1
            if cfg.verbose:
                elapsed = time.time() - start_time
                print(
                    f"[{completed}/{total_jobs}] Dataset={dataset_name} | "
                    f"Seed={seed} | Elapsed={elapsed:.1f}s"
                )

            run_rows, trace_rows = run_single_dataset_seed(
                dataset_name=dataset_name,
                x=x,
                y=y,
                seed=seed,
                cfg=cfg,
            )
            result_rows.extend(run_rows)
            selection_rows.extend(trace_rows)

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

    output = results.merge(lacp, on=["dataset", "seed"], how="left")
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
# Compact summaries
# ============================================================


SUMMARY_METRICS: Tuple[Tuple[str, str, float, str, int], ...] = (
    ("marginal_coverage", "Coverage", 1.0, "", 4),
    ("coverage_diff_vs_lacp", "Coverage Delta", 1.0, "", 4),
    ("width_change_vs_lacp", "Width Delta vs LACP", 100.0, "%", 2),
    ("score_change_vs_lacp", "Score Delta vs LACP", 100.0, "%", 2),
    ("q5_width_change_vs_lacp", "Weak-Q5 Width Delta", 100.0, "%", 2),
    ("q5_score_change_vs_lacp", "Weak-Q5 Score Delta", 100.0, "%", 2),
    ("epistemic_fraction", "Epistemic Fraction", 100.0, "%", 2),
    ("q5_epistemic_fraction", "Weak-Q5 Epistemic Fraction", 100.0, "%", 2),
)


def summarize_configurations(
    claps_results: pd.DataFrame,
    configuration_order: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    for configuration in configuration_order:
        subset = claps_results[
            claps_results["configuration"] == configuration
        ]
        if subset.empty:
            continue

        row: Dict[str, str] = {"Configuration": configuration}
        if configuration == "Selected":
            row["Prior Precision"] = "training-only selected"
        else:
            value = float(subset["prior_precision"].iloc[0])
            row["Prior Precision"] = lambda_label(value)

        for metric, label, multiplier, suffix, digits in SUMMARY_METRICS:
            values = subset[metric].to_numpy(dtype=float)
            mean_value = float(np.nanmean(values))
            se_value = safe_sem(values)
            row[label] = format_mean_se(
                mean_value,
                se_value,
                digits=digits,
                multiplier=multiplier,
                suffix=suffix,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_selection_frequency(
    selection_trace: pd.DataFrame,
    cfg: PriorPrecisionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = (
        selection_trace[selection_trace["is_selected"]]
        [["dataset", "seed", "prior_precision"]]
        .drop_duplicates()
    )

    total_runs = len(selected)
    overall_rows = []
    for lam in cfg.selection_lambda_grid:
        count = int(np.sum(np.isclose(selected["prior_precision"], lam)))
        overall_rows.append(
            {
                "Prior Precision": lambda_label(lam),
                "Count": count,
                "Selection Rate": count / max(total_runs, 1),
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall["Selection Rate"] = overall["Selection Rate"].map(
        lambda value: f"{100.0 * value:.2f}%"
    )

    dataset_rows = []
    for dataset_name in cfg.dataset_names:
        dataset_selected = selected[selected["dataset"] == dataset_name]
        denominator = len(dataset_selected)
        for lam in cfg.selection_lambda_grid:
            count = int(
                np.sum(np.isclose(dataset_selected["prior_precision"], lam))
            )
            dataset_rows.append(
                {
                    "Dataset": dataset_name,
                    "Prior Precision": lambda_label(lam),
                    "Count": count,
                    "Selection Rate": count / max(denominator, 1),
                }
            )

    dataset_frequency = pd.DataFrame(dataset_rows)
    dataset_frequency["Selection Rate"] = dataset_frequency[
        "Selection Rate"
    ].map(lambda value: f"{100.0 * value:.2f}%")
    return overall, dataset_frequency


def build_selection_nll_summary(
    selection_trace: pd.DataFrame,
    cfg: PriorPrecisionConfig,
) -> pd.DataFrame:
    rows = []
    for lam in cfg.selection_lambda_grid:
        subset = selection_trace[
            np.isclose(selection_trace["prior_precision"], lam)
        ]
        excess = subset["nll_excess_over_best"].to_numpy(dtype=float)
        nll = subset["validation_nll"].to_numpy(dtype=float)
        rows.append(
            {
                "Prior Precision": lambda_label(lam),
                "Validation NLL": format_mean_se(
                    float(np.mean(nll)),
                    safe_sem(nll),
                    digits=cfg.summary_digits,
                ),
                "NLL Excess over Best": format_mean_se(
                    float(np.mean(excess)),
                    safe_sem(excess),
                    digits=cfg.summary_digits,
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Hierarchical paired bootstrap for selected CLAPS
# ============================================================


def hierarchical_bootstrap_mean(
    data: pd.DataFrame,
    metric: str,
    cfg: PriorPrecisionConfig,
) -> Tuple[float, float, float]:
    datasets = data["dataset"].unique().tolist()
    if not datasets:
        raise ValueError("No datasets are available for bootstrap.")

    observed = float(np.mean(data[metric].to_numpy(dtype=float)))
    rng = np.random.default_rng(cfg.bootstrap_seed)
    boot = np.empty(cfg.bootstrap_repeats, dtype=float)

    dataset_values = {
        name: data.loc[data["dataset"] == name, metric].to_numpy(dtype=float)
        for name in datasets
    }

    for b in range(cfg.bootstrap_repeats):
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

        boot[b] = float(np.mean(sampled_values))

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return observed, float(ci_low), float(ci_high)


def build_selected_bootstrap_table(
    results: pd.DataFrame,
    cfg: PriorPrecisionConfig,
) -> pd.DataFrame:
    selected = results[
        (results["method"] == "CLAPS")
        & (results["selection_type"] == "Selected")
    ].copy()

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
    for metric, label, multiplier, suffix in metric_specs:
        mean_value, ci_low, ci_high = hierarchical_bootstrap_mean(
            selected,
            metric,
            cfg,
        )
        rows.append(
            {
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
    cfg: PriorPrecisionConfig,
) -> None:
    if not cfg.save_csv:
        return

    for suffix, frame in output_frames.items():
        path = f"{cfg.output_prefix}_{suffix}.csv"
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")


# ============================================================
# Run and display
# ============================================================


def main(cfg: PriorPrecisionConfig = CFG) -> Dict[str, pd.DataFrame]:
    prepare_packages()
    raw_results_df, selection_trace_df = run_experiment(cfg)
    results_df = add_lacp_relative_effects(raw_results_df)

    claps_df = results_df[results_df["method"] == "CLAPS"].copy()
    fixed_order = [
        f"lambda={lambda_label(value)}"
        for value in cfg.sensitivity_lambda_grid
    ]

    fixed_grid_summary_df = summarize_configurations(
        claps_results=claps_df[claps_df["selection_type"] == "Fixed"],
        configuration_order=fixed_order,
    )
    selected_summary_df = summarize_configurations(
        claps_results=claps_df[claps_df["selection_type"] == "Selected"],
        configuration_order=["Selected"],
    )

    (
        selection_frequency_df,
        dataset_selection_frequency_df,
    ) = build_selection_frequency(selection_trace_df, cfg)

    selection_nll_summary_df = build_selection_nll_summary(
        selection_trace_df,
        cfg,
    )
    selected_bootstrap_df = build_selected_bootstrap_table(results_df, cfg)

    output_frames = {
        "results": results_df,
        "selection_trace": selection_trace_df,
        "fixed_grid_summary": fixed_grid_summary_df,
        "selected_summary": selected_summary_df,
        "selection_frequency": selection_frequency_df,
        "dataset_selection_frequency": dataset_selection_frequency_df,
        "selection_nll_summary": selection_nll_summary_df,
        "selected_bootstrap": selected_bootstrap_df,
    }
    save_outputs(output_frames, cfg)

    print("\nFinished prior-precision selection and sensitivity analysis.")

    print("\nA. Fixed prior-precision sensitivity")
    display(fixed_grid_summary_df)

    print("\nB. Training-only selected prior precision")
    display(selected_summary_df)

    print("\nC. Overall selection frequency")
    display(selection_frequency_df)

    print("\nD. Selection NLL profile")
    display(selection_nll_summary_df)

    print("\nE. Hierarchical paired bootstrap: selected CLAPS versus LACP")
    display(selected_bootstrap_df)

    if cfg.display_raw_results:
        print("\nRaw per-dataset, per-seed results")
        display(results_df.round(cfg.summary_digits))

        print("\nDataset-specific selection frequency")
        display(dataset_selection_frequency_df)

    return output_frames


if __name__ == "__main__":
    OUTPUTS = main(CFG)


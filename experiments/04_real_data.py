
# ============================================================
# Experiment 4: Real-Data Benchmark and Support-Stratified Evaluation
# Single-cell Google Colab implementation
#
# Methods:
#   Split CP, CV+, LACP, CQR, UACQR-S, EPICSCORE-MDN, LWCP, CLAPS
#
# Main table reports exactly four metrics:
#   1) Marginal coverage
#   2) Width change relative to LACP
#   3) Interval-score change relative to LACP
#   4) Weakest learned-support-quintile score change relative to LACP
#
# The code additionally reports the paired CLAPS-LACP effect across
# raw-input and learned-feature support quintiles. DCP is omitted because
# the symmetric Gaussian DCP implementation in the original manuscript is
# algebraically identical to LACP.
# ============================================================

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# Package setup
# ============================================================

def install_if_missing(import_name: str, package_name: str = None) -> None:
    package_name = package_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )


install_if_missing("requests")
install_if_missing("openpyxl")
install_if_missing("xlrd")

import requests


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Final paper run: set num_seeds=20.
    # Five seeds keep the first Colab execution manageable.
    num_seeds: int = 20

    # Existing real-data protocol retained from the original manuscript.
    train_fraction: float = 0.60
    cal_fraction: float = 0.20
    test_fraction: float = 0.20
    max_samples_per_dataset: int = 5000

    # Nominal coverage.
    alpha: float = 0.10

    # Shared neural architecture.
    hidden_dim: int = 64
    feature_dim: int = 64
    quantile_dropout: float = 0.10
    epic_dropout: float = 0.15
    epic_components: int = 5

    # Optimization.
    max_epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5

    # CV+.
    cv_folds: int = 5

    # Last-layer Laplace.
    lambda_grid: Tuple[float, ...] = (
        1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
    )
    variance_floor: float = 1e-4
    jitter: float = 1e-6

    # UACQR-S: self-contained MC-dropout quantile-spread instantiation.
    uacqr_mc_samples: int = 40
    uacqr_spread_floor: float = 1e-3

    # EPICSCORE-MDN: self-contained posterior-predictive score-CDF instantiation.
    epic_fit_fraction: float = 0.50
    epic_mc_samples: int = 40
    epic_bisection_steps: int = 45
    epic_score_epsilon: float = 1e-6

    # LWCP: ridge-stabilized leverage in the shared learned representation.
    lwcp_ridge: float = 1e-3

    # Independent support proxies.
    support_k: int = 20
    support_quintiles: int = 5

    # Paired uncertainty summary.
    bootstrap_repeats: int = 5000

    # Display.
    verbose: bool = True
    summary_digits: int = 4
    display_full_support_results: bool = False


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("Experiment 4: real-data benchmark and support-stratified evaluation")
print("Main metrics: Coverage, Width Δ vs LACP, Score Δ vs LACP, Weak-Q5 Score Δ vs LACP")


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


def train_validation_split(
    n: int,
    validation_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.random.permutation(n)
    n_val = max(1, int(round(n * validation_fraction)))
    n_val = min(n_val, max(n - 1, 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


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


def lower_cv_quantile(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    k = int(np.floor((n + 1) * alpha))
    k = min(max(k, 1), n)
    return np.sort(values, axis=0)[k - 1, :]


def upper_cv_quantile(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return np.sort(values, axis=0)[k - 1, :]


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


def order_interval(
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return np.minimum(lower, upper), np.maximum(lower, upper)


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


def format_mean_se(mean_value: float, se_value: float, digits: int) -> str:
    if not np.isfinite(se_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}"


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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url, engine="xlrd")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_energy_efficiency() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    return clean_xy(df.iloc[:, :8], df.iloc[:, 8])


def load_yacht_hydrodynamics() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_airfoil_self_noise() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    df = pd.read_csv(url, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return clean_xy(df.iloc[:, :-1], df.iloc[:, -1])


def load_wine_quality_red() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return clean_xy(df.drop(columns=["quality"]), df["quality"])


def load_naval_propulsion() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/static/public/316/condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip"
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

    # Columns 0-15 are features; column 17 is GT turbine decay coefficient.
    return clean_xy(df.iloc[:, :16], df.iloc[:, 17])


def load_bike_sharing() -> Tuple[np.ndarray, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(BytesIO(response.content))
    df = pd.read_csv(archive.open("hour.csv"))

    target = df["cnt"]
    drop_cols = [
        col for col in ["cnt", "casual", "registered", "instant", "dteday"]
        if col in df.columns
    ]
    return clean_xy(df.drop(columns=drop_cols), target)


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing(as_frame=True)
    return clean_xy(data.data, data.target)


def load_all_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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

    datasets = {}
    for name, loader in loaders.items():
        print(f"Loading dataset: {name}")
        x, y = loader()
        if len(x) < 50:
            raise ValueError(f"Dataset {name} has too few usable rows.")
        datasets[name] = (x, y)
        print(f"  shape: X={x.shape}, y={y.shape}")
    return datasets


# ============================================================
# Split construction
# ============================================================

def make_random_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))

    if cfg.max_samples_per_dataset is not None and len(indices) > cfg.max_samples_per_dataset:
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
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]
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
# Neural models
# ============================================================

class MeanMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(feature_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.features(x))


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


class QuantileDropoutMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.quantile_head = nn.Linear(feature_dim, 2)

    def forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.quantile_head(self.backbone(x))
        return out[:, 0:1], out[:, 1:2]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qlo_raw, qhi_raw = self.forward_raw(x)
        return torch.minimum(qlo_raw, qhi_raw), torch.maximum(qlo_raw, qhi_raw)


class ScoreDistributionMLP(nn.Module):
    """MC-dropout mixture-density model for the conditional log-score CDF."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        n_components: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(hidden_dim, n_components)
        self.loc_head = nn.Linear(hidden_dim, n_components)
        self.scale_head = nn.Linear(hidden_dim, n_components)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.net(x)
        logits = self.logit_head(z)
        loc = self.loc_head(z)
        scale = F.softplus(self.scale_head(z)) + 1e-3
        return logits, loc, scale


# ============================================================
# Training functions
# ============================================================

def train_mean_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> MeanMLP:
    model = MeanMLP(x.shape[1], cfg.hidden_dim, cfg.feature_dim).to(DEVICE)
    x_t, y_t = to_tensor(x), to_tensor(y)
    train_idx, val_idx = train_validation_split(len(x), cfg.validation_fraction)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_state, best_val, wait = None, float("inf"), 0

    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_t[train_idx_t])
        loss = torch.mean((pred - y_t[train_idx_t]) ** 2)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_t[val_idx_t])
            val_loss = torch.mean((val_pred - y_t[val_idx_t]) ** 2).item()

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


def train_hetero_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> HeteroMLP:
    model = HeteroMLP(
        x.shape[1],
        cfg.hidden_dim,
        cfg.feature_dim,
        cfg.variance_floor,
    ).to(DEVICE)
    x_t, y_t = to_tensor(x), to_tensor(y)
    train_idx, val_idx = train_validation_split(len(x), cfg.validation_fraction)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_state, best_val, wait = None, float("inf"), 0

    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        mu, h2, _ = model(x_t[train_idx_t])
        residual2 = (y_t[train_idx_t] - mu) ** 2
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


def pinball_loss(y: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    diff = y - q
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))


def train_quantile_model(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> QuantileDropoutMLP:
    model = QuantileDropoutMLP(
        x.shape[1],
        cfg.hidden_dim,
        cfg.feature_dim,
        cfg.quantile_dropout,
    ).to(DEVICE)
    x_t, y_t = to_tensor(x), to_tensor(y)
    train_idx, val_idx = train_validation_split(len(x), cfg.validation_fraction)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    tau_lo = cfg.alpha / 2.0
    tau_hi = 1.0 - cfg.alpha / 2.0
    best_state, best_val, wait = None, float("inf"), 0

    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        qlo_raw, qhi_raw = model.forward_raw(x_t[train_idx_t])
        loss = (
            pinball_loss(y_t[train_idx_t], qlo_raw, tau_lo)
            + pinball_loss(y_t[train_idx_t], qhi_raw, tau_hi)
            + 0.1 * torch.mean(F.relu(qlo_raw - qhi_raw))
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_qlo, val_qhi = model.forward_raw(x_t[val_idx_t])
            val_loss = (
                pinball_loss(y_t[val_idx_t], val_qlo, tau_lo)
                + pinball_loss(y_t[val_idx_t], val_qhi, tau_hi)
                + 0.1 * torch.mean(F.relu(val_qlo - val_qhi))
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


def train_score_distribution_model(
    x: np.ndarray,
    log_score: np.ndarray,
    cfg: Config,
) -> ScoreDistributionMLP:
    model = ScoreDistributionMLP(
        x.shape[1],
        cfg.hidden_dim,
        cfg.epic_dropout,
        cfg.epic_components,
    ).to(DEVICE)
    x_t, score_t = to_tensor(x), to_tensor(log_score)
    train_idx, val_idx = train_validation_split(len(x), cfg.validation_fraction)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_state, best_val, wait = None, float("inf"), 0

    def mixture_nll(logits, loc, scale, target):
        target = target.expand_as(loc)
        standardized = (target - loc) / scale
        component_log_prob = (
            -0.5 * standardized ** 2
            - torch.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
            + F.log_softmax(logits, dim=-1)
        )
        return -torch.mean(torch.logsumexp(component_log_prob, dim=-1))

    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        logits, loc, scale = model(x_t[train_idx_t])
        loss = mixture_nll(logits, loc, scale, score_t[train_idx_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, loc, scale = model(x_t[val_idx_t])
            val_loss = mixture_nll(
                logits, loc, scale, score_t[val_idx_t]
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


# ============================================================
# Prediction helpers
# ============================================================

def predict_mean(model: MeanMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(to_tensor(x))
    return to_numpy(pred).reshape(-1, 1)


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


def predict_quantiles(
    model: QuantileDropoutMLP,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        qlo, qhi = model(to_tensor(x))
    return to_numpy(qlo).reshape(-1, 1), to_numpy(qhi).reshape(-1, 1)


def predict_quantiles_mc(
    model: QuantileDropoutMLP,
    x: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_t = to_tensor(x)
    lower_samples, upper_samples = [], []
    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            qlo, qhi = model(x_t)
            lower_samples.append(to_numpy(qlo).reshape(-1))
            upper_samples.append(to_numpy(qhi).reshape(-1))
    model.eval()
    return np.stack(lower_samples), np.stack(upper_samples)


def predict_score_distribution_mc(
    model: ScoreDistributionMLP,
    x: np.ndarray,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_t = to_tensor(x)
    logits_list, loc_list, scale_list = [], [], []
    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            logits, loc, scale = model(x_t)
            logits_list.append(logits)
            loc_list.append(loc)
            scale_list.append(scale)
    model.eval()
    return (
        torch.stack(logits_list),
        torch.stack(loc_list),
        torch.stack(scale_list),
    )


# ============================================================
# Last-layer geometry
# ============================================================

def add_bias_column(phi: np.ndarray) -> np.ndarray:
    ones = np.ones((len(phi), 1), dtype=np.float32)
    return np.concatenate([phi.astype(np.float32), ones], axis=1)


def compute_weighted_laplace_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    cfg: Config,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train).astype(np.float64)
    weights = 1.0 / np.maximum(
        h2_train.reshape(-1),
        cfg.variance_floor,
    )
    precision = prior_precision * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ (h_aug * weights[:, None])
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(precision)
    return covariance


def compute_unweighted_feature_covariance(
    phi_train: np.ndarray,
    ridge: float,
    cfg: Config,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train).astype(np.float64)
    precision = ridge * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ h_aug
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)
    try:
        return np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(precision)


def quadratic_form(phi: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    h_aug = add_bias_column(phi).astype(np.float64)
    value = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(value, 0.0).astype(np.float32)


def choose_prior_precision(
    model: HeteroMLP,
    x_train: np.ndarray,
    y_train_std: np.ndarray,
    cfg: Config,
) -> float:
    fit_idx, val_idx = train_validation_split(
        len(x_train),
        cfg.validation_fraction,
    )
    _, h2_fit, phi_fit = predict_hetero(model, x_train[fit_idx])
    mu_val, h2_val, phi_val = predict_hetero(model, x_train[val_idx])
    residual2 = (y_train_std[val_idx] - mu_val) ** 2

    best_lambda = float(cfg.lambda_grid[0])
    best_nll = float("inf")
    for lam in cfg.lambda_grid:
        covariance = compute_weighted_laplace_covariance(
            phi_fit,
            h2_fit,
            lam,
            cfg,
        )
        epi_val = quadratic_form(phi_val, covariance)
        total_var = np.maximum(h2_val + epi_val, cfg.variance_floor)
        nll = 0.5 * np.mean(residual2 / total_var + np.log(total_var))
        if nll < best_nll:
            best_nll = float(nll)
            best_lambda = float(lam)
    return best_lambda


# ============================================================
# Conformal methods
# ============================================================

def run_split_cp(
    model: MeanMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_cal = predict_mean(model, data["x_cal"])
    mu_test = predict_mean(model, data["x_test"])
    qhat = finite_sample_quantile(
        np.abs(data["y_cal_std"] - mu_cal),
        cfg.alpha,
    )
    return (
        unstandardize_y(mu_test - qhat, data["y_mean"], data["y_scale"]),
        unstandardize_y(mu_test + qhat, data["y_mean"], data["y_scale"]),
    )


def run_cv_plus(
    x_all: np.ndarray,
    y_all_std: np.ndarray,
    x_test: np.ndarray,
    y_mean: float,
    y_scale: float,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_splits = min(cfg.cv_folds, len(x_all))
    if n_splits < 2:
        raise ValueError("CV+ requires at least two folds.")

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=10000 + seed,
    )
    fold_ids = np.empty(len(x_all), dtype=int)
    residuals = np.empty(len(x_all), dtype=np.float32)
    test_predictions = []

    for fold_id, (fit_idx, holdout_idx) in enumerate(kf.split(x_all)):
        model = train_mean_model(x_all[fit_idx], y_all_std[fit_idx], cfg)
        holdout_pred = predict_mean(model, x_all[holdout_idx]).reshape(-1)
        residuals[holdout_idx] = np.abs(
            y_all_std[holdout_idx].reshape(-1) - holdout_pred
        )
        fold_ids[holdout_idx] = fold_id
        test_predictions.append(predict_mean(model, x_test).reshape(-1))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_predictions = np.stack(test_predictions)
    prediction_for_point = test_predictions[fold_ids, :]
    lower_candidates = prediction_for_point - residuals[:, None]
    upper_candidates = prediction_for_point + residuals[:, None]
    lower_std = lower_cv_quantile(lower_candidates, cfg.alpha).reshape(-1, 1)
    upper_std = upper_cv_quantile(upper_candidates, cfg.alpha).reshape(-1, 1)
    return (
        unstandardize_y(lower_std, y_mean, y_scale),
        unstandardize_y(upper_std, y_mean, y_scale),
    )


def run_lacp(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_cal, h2_cal, _ = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, _ = predict_hetero(model, data["x_test"])
    h_cal = np.sqrt(np.maximum(h2_cal, cfg.variance_floor))
    h_test = np.sqrt(np.maximum(h2_test, cfg.variance_floor))
    qhat = finite_sample_quantile(
        np.abs(data["y_cal_std"] - mu_cal) / h_cal,
        cfg.alpha,
    )
    return (
        unstandardize_y(mu_test - qhat * h_test, data["y_mean"], data["y_scale"]),
        unstandardize_y(mu_test + qhat * h_test, data["y_mean"], data["y_scale"]),
    )


def run_cqr(
    model: QuantileDropoutMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    qlo_cal, qhi_cal = predict_quantiles(model, data["x_cal"])
    qlo_test, qhi_test = predict_quantiles(model, data["x_test"])
    scores = np.maximum(
        qlo_cal - data["y_cal_std"],
        data["y_cal_std"] - qhi_cal,
    )
    qhat = finite_sample_quantile(scores, cfg.alpha)
    lower_std, upper_std = order_interval(
        qlo_test - qhat,
        qhi_test + qhat,
    )
    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def run_uacqr_s(
    model: QuantileDropoutMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    qlo_cal, qhi_cal = predict_quantiles(model, data["x_cal"])
    qlo_test, qhi_test = predict_quantiles(model, data["x_test"])

    qlo_cal_mc, qhi_cal_mc = predict_quantiles_mc(
        model,
        data["x_cal"],
        cfg.uacqr_mc_samples,
    )
    qlo_test_mc, qhi_test_mc = predict_quantiles_mc(
        model,
        data["x_test"],
        cfg.uacqr_mc_samples,
    )

    spread_lo_cal = np.maximum(
        np.std(qlo_cal_mc, axis=0, ddof=1).reshape(-1, 1),
        cfg.uacqr_spread_floor,
    )
    spread_hi_cal = np.maximum(
        np.std(qhi_cal_mc, axis=0, ddof=1).reshape(-1, 1),
        cfg.uacqr_spread_floor,
    )
    spread_lo_test = np.maximum(
        np.std(qlo_test_mc, axis=0, ddof=1).reshape(-1, 1),
        cfg.uacqr_spread_floor,
    )
    spread_hi_test = np.maximum(
        np.std(qhi_test_mc, axis=0, ddof=1).reshape(-1, 1),
        cfg.uacqr_spread_floor,
    )

    scores = np.maximum(
        (qlo_cal - data["y_cal_std"]) / spread_lo_cal,
        (data["y_cal_std"] - qhi_cal) / spread_hi_cal,
    )
    qhat = finite_sample_quantile(scores, cfg.alpha)
    lower_std, upper_std = order_interval(
        qlo_test - qhat * spread_lo_test,
        qhi_test + qhat * spread_hi_test,
    )
    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def mixture_normal_cdf(
    z: torch.Tensor,
    logit_samples: torch.Tensor,
    loc_samples: torch.Tensor,
    scale_samples: torch.Tensor,
) -> torch.Tensor:
    standardized = (z.unsqueeze(0).unsqueeze(-1) - loc_samples) / scale_samples
    component_cdf = 0.5 * (
        1.0 + torch.erf(standardized / math.sqrt(2.0))
    )
    weights = torch.softmax(logit_samples, dim=-1)
    draw_cdf = torch.sum(weights * component_cdf, dim=-1)
    return torch.mean(draw_cdf, dim=0)


def invert_mixture_normal_cdf(
    target_probability: float,
    logit_samples: torch.Tensor,
    loc_samples: torch.Tensor,
    scale_samples: torch.Tensor,
    steps: int,
) -> np.ndarray:
    target = float(np.clip(target_probability, 1e-6, 1.0 - 1e-6))
    lower = torch.amin(loc_samples - 8.0 * scale_samples, dim=(0, 2))
    upper = torch.amax(loc_samples + 8.0 * scale_samples, dim=(0, 2))

    for _ in range(steps):
        midpoint = 0.5 * (lower + upper)
        cdf_mid = mixture_normal_cdf(
            midpoint,
            logit_samples,
            loc_samples,
            scale_samples,
        )
        lower = torch.where(cdf_mid < target, midpoint, lower)
        upper = torch.where(cdf_mid >= target, midpoint, upper)
    return to_numpy(0.5 * (lower + upper)).reshape(-1, 1)


def run_epicscore(
    hetero_model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_cal, h2_cal, _ = predict_hetero(hetero_model, data["x_cal"])
    mu_test, h2_test, _ = predict_hetero(hetero_model, data["x_test"])
    h_cal = np.sqrt(np.maximum(h2_cal, cfg.variance_floor))
    h_test = np.sqrt(np.maximum(h2_test, cfg.variance_floor))
    base_scores = np.abs(data["y_cal_std"] - mu_cal) / h_cal

    idx = np.random.permutation(len(data["x_cal"]))
    n_fit = int(round(cfg.epic_fit_fraction * len(idx)))
    n_fit = min(max(n_fit, 15), len(idx) - 15)
    if n_fit < 2 or len(idx) - n_fit < 2:
        raise ValueError("Calibration set is too small for EPICSCORE-MDN.")
    fit_idx, conformal_idx = idx[:n_fit], idx[n_fit:]

    log_scores_fit = np.log(
        base_scores[fit_idx] + cfg.epic_score_epsilon
    ).astype(np.float32)
    score_model = train_score_distribution_model(
        data["x_cal"][fit_idx],
        log_scores_fit,
        cfg,
    )

    logits_cal, loc_cal, scale_cal = predict_score_distribution_mc(
        score_model,
        data["x_cal"][conformal_idx],
        cfg.epic_mc_samples,
    )
    log_scores_cal = torch.log(
        to_tensor(base_scores[conformal_idx].reshape(-1))
        + cfg.epic_score_epsilon
    )
    transformed = mixture_normal_cdf(
        log_scores_cal,
        logits_cal,
        loc_cal,
        scale_cal,
    )
    threshold_probability = finite_sample_quantile(
        to_numpy(transformed),
        cfg.alpha,
    )

    logits_test, loc_test, scale_test = predict_score_distribution_mc(
        score_model,
        data["x_test"],
        cfg.epic_mc_samples,
    )
    log_threshold = invert_mixture_normal_cdf(
        threshold_probability,
        logits_test,
        loc_test,
        scale_test,
        cfg.epic_bisection_steps,
    )
    score_threshold = np.maximum(
        np.exp(log_threshold) - cfg.epic_score_epsilon,
        0.0,
    )
    lower_std = mu_test - score_threshold * h_test
    upper_std = mu_test + score_threshold * h_test

    del score_model, logits_cal, loc_cal, scale_cal
    del logits_test, loc_test, scale_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def run_lwcp(
    hetero_model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, phi_train = predict_hetero(hetero_model, data["x_train"])
    mu_cal, _, phi_cal = predict_hetero(hetero_model, data["x_cal"])
    mu_test, _, phi_test = predict_hetero(hetero_model, data["x_test"])

    covariance = compute_unweighted_feature_covariance(
        phi_train,
        cfg.lwcp_ridge,
        cfg,
    )
    leverage_cal = quadratic_form(phi_cal, covariance)
    leverage_test = quadratic_form(phi_test, covariance)
    weight_cal = 1.0 / np.sqrt(1.0 + leverage_cal)
    weight_test = 1.0 / np.sqrt(1.0 + leverage_test)

    qhat = finite_sample_quantile(
        np.abs(data["y_cal_std"] - mu_cal) * weight_cal,
        cfg.alpha,
    )
    half_width = qhat / np.maximum(weight_test, 1e-8)
    return (
        unstandardize_y(mu_test - half_width, data["y_mean"], data["y_scale"]),
        unstandardize_y(mu_test + half_width, data["y_mean"], data["y_scale"]),
    )


def run_claps(
    hetero_model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    chosen_lambda = choose_prior_precision(
        hetero_model,
        data["x_train"],
        data["y_train_std"],
        cfg,
    )
    _, h2_train, phi_train = predict_hetero(
        hetero_model,
        data["x_train"],
    )
    mu_cal, h2_cal, phi_cal = predict_hetero(
        hetero_model,
        data["x_cal"],
    )
    mu_test, h2_test, phi_test = predict_hetero(
        hetero_model,
        data["x_test"],
    )

    covariance = compute_weighted_laplace_covariance(
        phi_train,
        h2_train,
        chosen_lambda,
        cfg,
    )
    epi_cal = quadratic_form(phi_cal, covariance)
    epi_test = quadratic_form(phi_test, covariance)
    total_var_cal = np.maximum(h2_cal + epi_cal, cfg.variance_floor)
    total_var_test = np.maximum(h2_test + epi_test, cfg.variance_floor)

    qhat = finite_sample_quantile(
        np.abs(data["y_cal_std"] - mu_cal) / np.sqrt(total_var_cal),
        cfg.alpha,
    )
    scale_test = np.sqrt(total_var_test)
    return (
        unstandardize_y(mu_test - qhat * scale_test, data["y_mean"], data["y_scale"]),
        unstandardize_y(mu_test + qhat * scale_test, data["y_mean"], data["y_scale"]),
    )


# ============================================================
# Independent support proxies
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
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(train_features)
    distances, _ = model.kneighbors(test_features, return_distance=True)
    return np.mean(distances, axis=1)


def rank_based_quintiles(
    values: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    groups = np.empty(len(values), dtype=int)
    # Q1 = strongest support; Q5 = weakest support.
    groups[order] = (
        np.floor(np.arange(len(values)) * n_groups / len(values)).astype(int) + 1
    )
    return np.minimum(groups, n_groups)


def compute_support_groups(
    hetero_model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Dict[str, np.ndarray]:
    raw_distance = knn_support_distance(
        data["x_train"],
        data["x_test"],
        cfg.support_k,
    )

    _, _, phi_train = predict_hetero(hetero_model, data["x_train"])
    _, _, phi_test = predict_hetero(hetero_model, data["x_test"])
    phi_train_std, phi_test_std = standardize_feature_space(
        phi_train,
        phi_test,
    )
    feature_distance = knn_support_distance(
        phi_train_std,
        phi_test_std,
        cfg.support_k,
    )

    return {
        "Raw input": rank_based_quintiles(
            raw_distance,
            cfg.support_quintiles,
        ),
        "Learned feature": rank_based_quintiles(
            feature_distance,
            cfg.support_quintiles,
        ),
    }


# ============================================================
# Metrics
# ============================================================

def compute_run_metrics(
    dataset_name: str,
    method_name: str,
    seed: int,
    data: Dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    support_groups: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    y = data["y_test"].reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    scores = interval_score(y, lower, upper, cfg.alpha)
    y_scale = max(float(data["y_scale"]), 1e-12)

    feature_q = support_groups["Learned feature"]
    weak_mask = feature_q == cfg.support_quintiles

    overall_row = {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "marginal_coverage": float(np.mean(covered)),
        "normalized_width": float(np.mean(width) / y_scale),
        "normalized_interval_score": float(np.mean(scores) / y_scale),
        "weak_feature_interval_score": float(
            np.mean(scores[weak_mask]) / y_scale
        ),
    }

    support_rows = []
    for support_space, groups in support_groups.items():
        for quintile in range(1, cfg.support_quintiles + 1):
            mask = groups == quintile
            support_rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "seed": seed,
                "support_space": support_space,
                "quintile": quintile,
                "coverage": float(np.mean(covered[mask])),
                "normalized_width": float(np.mean(width[mask]) / y_scale),
                "normalized_interval_score": float(
                    np.mean(scores[mask]) / y_scale
                ),
            })

    return overall_row, support_rows


# ============================================================
# One dataset-seed run
# ============================================================

def run_single_dataset_seed(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    set_seed(seed)
    data = make_random_split(x, y, seed, cfg)

    mean_model = train_mean_model(
        data["x_train"],
        data["y_train_std"],
        cfg,
    )
    hetero_model = train_hetero_model(
        data["x_train"],
        data["y_train_std"],
        cfg,
    )
    quantile_model = train_quantile_model(
        data["x_train"],
        data["y_train_std"],
        cfg,
    )

    support_groups = compute_support_groups(hetero_model, data, cfg)

    x_all = np.concatenate([data["x_train"], data["x_cal"]], axis=0)
    y_all_std = np.concatenate(
        [data["y_train_std"], data["y_cal_std"]],
        axis=0,
    )

    method_runners = [
        ("Split CP", lambda: run_split_cp(mean_model, data, cfg)),
        (
            "CV+",
            lambda: run_cv_plus(
                x_all=x_all,
                y_all_std=y_all_std,
                x_test=data["x_test"],
                y_mean=data["y_mean"],
                y_scale=data["y_scale"],
                cfg=cfg,
                seed=seed,
            ),
        ),
        ("LACP", lambda: run_lacp(hetero_model, data, cfg)),
        ("CQR", lambda: run_cqr(quantile_model, data, cfg)),
        ("UACQR-S", lambda: run_uacqr_s(quantile_model, data, cfg)),
        ("EPICSCORE-MDN", lambda: run_epicscore(hetero_model, data, cfg)),
        ("LWCP", lambda: run_lwcp(hetero_model, data, cfg)),
        ("CLAPS", lambda: run_claps(hetero_model, data, cfg)),
    ]

    overall_rows, support_rows = [], []
    for method_name, runner in method_runners:
        lower, upper = runner()
        overall_row, method_support_rows = compute_run_metrics(
            dataset_name,
            method_name,
            seed,
            data,
            lower,
            upper,
            support_groups,
            cfg,
        )
        overall_rows.append(overall_row)
        support_rows.extend(method_support_rows)

    del mean_model, hetero_model, quantile_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return overall_rows, support_rows


# ============================================================
# Main loop
# ============================================================

def run_experiment(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = load_all_datasets()
    overall_rows, support_rows = [], []
    total_jobs = len(datasets) * cfg.num_seeds
    completed = 0
    start_time = time.time()

    for dataset_name, (x, y) in datasets.items():
        for seed in range(cfg.num_seeds):
            completed += 1
            if cfg.verbose:
                elapsed = time.time() - start_time
                print(
                    f"[{completed}/{total_jobs}] "
                    f"Dataset={dataset_name} | Seed={seed} | "
                    f"Elapsed={elapsed:.1f}s"
                )

            seed_overall, seed_support = run_single_dataset_seed(
                dataset_name,
                x,
                y,
                seed,
                cfg,
            )
            overall_rows.extend(seed_overall)
            support_rows.extend(seed_support)

    return pd.DataFrame(overall_rows), pd.DataFrame(support_rows)


# ============================================================
# Relative effects and compact summaries
# ============================================================

def add_lacp_relative_effects(
    overall_df: pd.DataFrame,
    support_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lacp_overall = (
        overall_df[overall_df["method"] == "LACP"]
        [[
            "dataset",
            "seed",
            "normalized_width",
            "normalized_interval_score",
            "weak_feature_interval_score",
        ]]
        .rename(columns={
            "normalized_width": "lacp_width",
            "normalized_interval_score": "lacp_score",
            "weak_feature_interval_score": "lacp_weak_score",
        })
    )
    overall_df = overall_df.merge(
        lacp_overall,
        on=["dataset", "seed"],
        how="left",
    )
    overall_df["width_change_vs_lacp"] = (
        overall_df["normalized_width"] / overall_df["lacp_width"] - 1.0
    )
    overall_df["score_change_vs_lacp"] = (
        overall_df["normalized_interval_score"] / overall_df["lacp_score"] - 1.0
    )
    overall_df["weak_score_change_vs_lacp"] = (
        overall_df["weak_feature_interval_score"]
        / overall_df["lacp_weak_score"]
        - 1.0
    )

    lacp_support = (
        support_df[support_df["method"] == "LACP"]
        [[
            "dataset",
            "seed",
            "support_space",
            "quintile",
            "coverage",
            "normalized_width",
            "normalized_interval_score",
        ]]
        .rename(columns={
            "coverage": "lacp_coverage",
            "normalized_width": "lacp_width",
            "normalized_interval_score": "lacp_score",
        })
    )
    support_df = support_df.merge(
        lacp_support,
        on=["dataset", "seed", "support_space", "quintile"],
        how="left",
    )
    support_df["coverage_diff_vs_lacp"] = (
        support_df["coverage"] - support_df["lacp_coverage"]
    )
    support_df["width_change_vs_lacp"] = (
        support_df["normalized_width"] / support_df["lacp_width"] - 1.0
    )
    support_df["score_change_vs_lacp"] = (
        support_df["normalized_interval_score"] / support_df["lacp_score"] - 1.0
    )
    return overall_df, support_df


def build_main_summary(
    overall_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    method_order = [
        "Split CP",
        "CV+",
        "LACP",
        "CQR",
        "UACQR-S",
        "EPICSCORE-MDN",
        "LWCP",
        "CLAPS",
    ]
    metrics = [
        "marginal_coverage",
        "width_change_vs_lacp",
        "score_change_vs_lacp",
        "weak_score_change_vs_lacp",
    ]
    grouped = overall_df.groupby("method")[metrics].agg(["mean", "sem"])

    rows = []
    for method in method_order:
        if method not in grouped.index:
            continue
        rows.append({
            "Method": method,
            "Coverage": format_mean_se(
                grouped.loc[method, ("marginal_coverage", "mean")],
                grouped.loc[method, ("marginal_coverage", "sem")],
                cfg.summary_digits,
            ),
            "Width Δ vs LACP": format_mean_se(
                100.0 * grouped.loc[method, ("width_change_vs_lacp", "mean")],
                100.0 * grouped.loc[method, ("width_change_vs_lacp", "sem")],
                2,
            ) + "%",
            "Score Δ vs LACP": format_mean_se(
                100.0 * grouped.loc[method, ("score_change_vs_lacp", "mean")],
                100.0 * grouped.loc[method, ("score_change_vs_lacp", "sem")],
                2,
            ) + "%",
            "Weak-Q5 Score Δ": format_mean_se(
                100.0 * grouped.loc[method, ("weak_score_change_vs_lacp", "mean")],
                100.0 * grouped.loc[method, ("weak_score_change_vs_lacp", "sem")],
                2,
            ) + "%",
        })
    return pd.DataFrame(rows)


def build_claps_support_summary(
    support_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    claps = support_df[support_df["method"] == "CLAPS"].copy()
    grouped = (
        claps
        .groupby(["support_space", "quintile"])[
            [
                "coverage_diff_vs_lacp",
                "width_change_vs_lacp",
                "score_change_vs_lacp",
            ]
        ]
        .agg(["mean", "sem"])
    )

    rows = []
    for support_space in ["Raw input", "Learned feature"]:
        for quintile in range(1, cfg.support_quintiles + 1):
            key = (support_space, quintile)
            if key not in grouped.index:
                continue
            rows.append({
                "Support": support_space,
                "Quintile": f"Q{quintile}",
                "Coverage Δ": format_mean_se(
                    grouped.loc[key, ("coverage_diff_vs_lacp", "mean")],
                    grouped.loc[key, ("coverage_diff_vs_lacp", "sem")],
                    cfg.summary_digits,
                ),
                "Width Δ": format_mean_se(
                    100.0 * grouped.loc[key, ("width_change_vs_lacp", "mean")],
                    100.0 * grouped.loc[key, ("width_change_vs_lacp", "sem")],
                    2,
                ) + "%",
                "Score Δ": format_mean_se(
                    100.0 * grouped.loc[key, ("score_change_vs_lacp", "mean")],
                    100.0 * grouped.loc[key, ("score_change_vs_lacp", "sem")],
                    2,
                ) + "%",
            })
    return pd.DataFrame(rows)


def hierarchical_paired_bootstrap(
    overall_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    claps = overall_df[overall_df["method"] == "CLAPS"].copy()
    lacp = overall_df[overall_df["method"] == "LACP"].copy()
    paired = claps.merge(
        lacp,
        on=["dataset", "seed"],
        suffixes=("_claps", "_lacp"),
    )

    paired["coverage_diff"] = (
        paired["marginal_coverage_claps"]
        - paired["marginal_coverage_lacp"]
    )
    paired["width_change"] = (
        paired["normalized_width_claps"]
        / paired["normalized_width_lacp"]
        - 1.0
    )
    paired["score_change"] = (
        paired["normalized_interval_score_claps"]
        / paired["normalized_interval_score_lacp"]
        - 1.0
    )
    paired["weak_score_change"] = (
        paired["weak_feature_interval_score_claps"]
        / paired["weak_feature_interval_score_lacp"]
        - 1.0
    )

    metric_specs = [
        ("coverage_diff", "Coverage Δ", 1.0),
        ("width_change", "Width Δ vs LACP", 100.0),
        ("score_change", "Score Δ vs LACP", 100.0),
        ("weak_score_change", "Weak-Q5 Score Δ", 100.0),
    ]
    datasets = paired["dataset"].unique().tolist()
    rng = np.random.default_rng(20260731)
    output_rows = []

    for metric, label, multiplier in metric_specs:
        observed = float(np.mean(paired[metric])) * multiplier
        boot = np.empty(cfg.bootstrap_repeats, dtype=float)

        for b in range(cfg.bootstrap_repeats):
            sampled_datasets = rng.choice(
                datasets,
                size=len(datasets),
                replace=True,
            )
            sampled_values = []
            for dataset_name in sampled_datasets:
                dataset_values = paired.loc[
                    paired["dataset"] == dataset_name,
                    metric,
                ].to_numpy()
                sampled_values.extend(
                    rng.choice(
                        dataset_values,
                        size=len(dataset_values),
                        replace=True,
                    )
                )
            boot[b] = float(np.mean(sampled_values)) * multiplier

        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        suffix = "%" if multiplier == 100.0 else ""
        output_rows.append({
            "Effect": label,
            "Mean": f"{observed:.4f}{suffix}",
            "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]{suffix}",
        })

    return pd.DataFrame(output_rows)


# ============================================================
# Run and display
# ============================================================

overall_results_df, support_results_df = run_experiment(CFG)
overall_results_df, support_results_df = add_lacp_relative_effects(
    overall_results_df,
    support_results_df,
)

main_summary_df = build_main_summary(overall_results_df, CFG)
claps_support_summary_df = build_claps_support_summary(
    support_results_df,
    CFG,
)
paired_claps_lacp_df = hierarchical_paired_bootstrap(
    overall_results_df,
    CFG,
)

print("\nFinished Experiment 4.")
print("\nMain real-data benchmark summary:")
display(main_summary_df)

print("\nCLAPS - LACP by independent support quintile:")
print("Q1 is strongest support; Q5 is weakest support.")
display(claps_support_summary_df)

print("\nHierarchical paired bootstrap: CLAPS versus LACP")
display(paired_claps_lacp_df)

if CFG.display_full_support_results:
    print("\nFull per-dataset, per-seed, per-method support results:")
    display(support_results_df.round(CFG.summary_digits))


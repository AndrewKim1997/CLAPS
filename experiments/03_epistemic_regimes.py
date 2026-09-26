
# ============================================================
# Experiment 3: When Does the Epistemic Correction Matter?
# Single-cell Google Colab implementation
#
# This experiment extends the former posterior-contraction study.
# It contains two controlled sweeps:
#   A) training-size sweep at fixed weak local support
#   B) local-support sweep at fixed total training size
#
# The shared heteroscedastic neural model is fitted once per
# setting and used by all four methods:
#   1) LACP
#   2) Unweighted Laplace
#   3) LWCP
#   4) CLAPS
#
# Only four quantities are reported:
#   1) Marginal coverage
#   2) Sparse / dense-low-noise width ratio
#   3) Sparse-region interval score
#   4) CLAPS epistemic fraction
# ============================================================

import gc
import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import display

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Final paper run: set num_seeds=30.
    # Five seeds keep the first Colab run manageable.
    num_seeds: int = 30

    # Sweep A: posterior contraction as total training information grows.
    n_train_grid: Tuple[int, ...] = (100, 250, 500, 1000, 2000)
    fixed_sparse_retention: float = 0.10

    # Sweep B: local-support severity at fixed total training size.
    fixed_n_train: int = 1000
    sparse_retention_grid: Tuple[float, ...] = (
        0.01, 0.025, 0.05, 0.10, 0.20, 1.00
    )

    # Calibration and test sizes
    n_cal: int = 2000
    n_test: int = 10000

    # Ten-dimensional target distribution:
    # dense-low-noise, sparse-low-noise, dense-high-noise.
    input_dim: int = 10
    target_region_probs: Tuple[float, float, float] = (0.40, 0.35, 0.25)

    # Data-generating process
    cluster_std: float = 0.75
    low_noise_std: float = 0.15
    high_noise_std: float = 0.65
    sparse_bump_amplitude: float = 1.00
    sparse_bump_bandwidth: float = 0.80

    # Nominal coverage
    alpha: float = 0.10

    # Shared heteroscedastic neural model
    hidden_dim: int = 64
    feature_dim: int = 64
    max_epochs: int = 220
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5
    variance_floor: float = 1e-4

    # Last-layer posterior
    lambda_grid: Tuple[float, ...] = (
        1e-5, 1e-4, 1e-3, 1e-2,
        1e-1, 1.0, 10.0, 100.0
    )
    jitter: float = 1e-6

    # LWCP ridge parameter
    lwcp_ridge: float = 1e-3

    # Display
    verbose: bool = True
    summary_digits: int = 4
    display_raw_results: bool = False


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGION_DENSE_LOW = 0
REGION_SPARSE_LOW = 1
REGION_DENSE_HIGH = 2

METHOD_ORDER = [
    "LACP",
    "Unweighted Laplace",
    "LWCP",
    "CLAPS",
]

print(f"Using device: {DEVICE}")
print("Experiment 3: when does the epistemic correction matter?")
print(
    "Reported quantities: Marginal Cov., Sparse/Dense Width, "
    "Sparse Int. Score, CLAPS Epi. Frac."
)


# ============================================================
# Generic utilities
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
    n_val = min(n_val, n - 1) if n > 1 else 1

    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]

    if len(fit_idx) == 0:
        fit_idx = val_idx

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


def format_mean_se(mean_value: float, se_value: float, digits: int) -> str:
    if not np.isfinite(mean_value):
        return "—"
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}"


# ============================================================
# Ten-dimensional weak-support data-generating process
# ============================================================

def region_centers(input_dim: int) -> np.ndarray:
    centers = np.zeros((3, input_dim), dtype=np.float32)

    # The first two coordinates determine the three regions.
    # The remaining coordinates are nuisance dimensions.
    centers[REGION_DENSE_LOW, 0:2] = np.array([-2.5, 0.0])
    centers[REGION_SPARSE_LOW, 0:2] = np.array([2.5, 0.0])
    centers[REGION_DENSE_HIGH, 0:2] = np.array([0.0, 2.8])

    return centers


def sample_target_mixture(
    n: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(cfg.target_region_probs, dtype=float)
    probs = probs / probs.sum()

    region = np.random.choice(3, size=n, p=probs)
    centers = region_centers(cfg.input_dim)
    x = centers[region] + cfg.cluster_std * np.random.randn(n, cfg.input_dim)

    return x.astype(np.float32), region.astype(np.int64)


def sample_training_mixture(
    n: int,
    sparse_retention: float,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from the target mixture, then thin only the sparse region.

    sparse_retention=1.0 reproduces the target support distribution.
    Smaller values reduce local training support without changing the
    calibration or test distribution.
    """
    if not (0.0 < sparse_retention <= 1.0):
        raise ValueError("sparse_retention must lie in (0, 1].")

    x_batches: List[np.ndarray] = []
    region_batches: List[np.ndarray] = []
    accepted_count = 0

    while accepted_count < n:
        remaining = n - accepted_count
        batch_size = max(1000, 3 * remaining)

        x_candidate, region_candidate = sample_target_mixture(batch_size, cfg)
        accept_prob = np.ones(batch_size, dtype=np.float32)
        accept_prob[region_candidate == REGION_SPARSE_LOW] = sparse_retention
        accept = np.random.rand(batch_size) < accept_prob

        accepted_x = x_candidate[accept]
        accepted_region = region_candidate[accept]

        x_batches.append(accepted_x)
        region_batches.append(accepted_region)
        accepted_count += len(accepted_x)

    x = np.concatenate(x_batches, axis=0)[:n]
    region = np.concatenate(region_batches, axis=0)[:n]

    return x.astype(np.float32), region.astype(np.int64)


def true_mean(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    base = (
        0.80 * np.sin(x[:, 0:1])
        + 0.35 * x[:, 1:2]
        + 0.20 * np.square(x[:, 2:3])
        - 0.18 * x[:, 0:1] * x[:, 1:2]
        + 0.15 * np.tanh(x[:, 3:4])
    )

    sparse_center = region_centers(cfg.input_dim)[REGION_SPARSE_LOW]
    distance2 = np.sum(
        np.square(x[:, 0:2] - sparse_center[None, 0:2]),
        axis=1,
        keepdims=True,
    )

    local_bump = cfg.sparse_bump_amplitude * np.exp(
        -distance2 / (2.0 * cfg.sparse_bump_bandwidth ** 2)
    )

    return (base + local_bump).astype(np.float32)


def true_noise_std(region: np.ndarray, cfg: Config) -> np.ndarray:
    sigma = np.full((len(region), 1), cfg.low_noise_std, dtype=np.float32)
    sigma[region == REGION_DENSE_HIGH] = cfg.high_noise_std
    return sigma


def generate_y(
    x: np.ndarray,
    region: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    mean = true_mean(x, cfg)
    sigma = true_noise_std(region, cfg)
    y = mean + sigma * np.random.randn(len(x), 1)
    return y.astype(np.float32)


def make_dataset(
    n_train: int,
    sparse_retention: float,
    seed: int,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    # The setting-specific seed makes repeated runs reproducible and ensures
    # that an identical (n_train, retention, seed) setting is identical in
    # both sweeps.
    setting_seed = (
        2_000_000
        + 10_000 * int(seed)
        + 7 * int(n_train)
        + int(round(10_000 * sparse_retention))
    )
    set_seed(setting_seed)

    x_train_raw, region_train = sample_training_mixture(
        n_train,
        sparse_retention,
        cfg,
    )
    x_cal_raw, region_cal = sample_target_mixture(cfg.n_cal, cfg)
    x_test_raw, region_test = sample_target_mixture(cfg.n_test, cfg)

    y_train = generate_y(x_train_raw, region_train, cfg)
    y_cal = generate_y(x_cal_raw, region_cal, cfg)
    y_test = generate_y(x_test_raw, region_test, cfg)

    # All standardization statistics are fitted on the training set only.
    x_mean = np.mean(x_train_raw, axis=0, keepdims=True)
    x_scale = np.std(x_train_raw, axis=0, keepdims=True) + 1e-6

    x_train = ((x_train_raw - x_mean) / x_scale).astype(np.float32)
    x_cal = ((x_cal_raw - x_mean) / x_scale).astype(np.float32)
    x_test = ((x_test_raw - x_mean) / x_scale).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train) + 1e-8)

    return {
        "x_train": x_train,
        "x_cal": x_cal,
        "x_test": x_test,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
        "y_train_std": ((y_train - y_mean) / y_scale).astype(np.float32),
        "y_cal_std": ((y_cal - y_mean) / y_scale).astype(np.float32),
        "region_train": region_train,
        "region_cal": region_cal,
        "region_test": region_test,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


# ============================================================
# Shared heteroscedastic neural model
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
    cfg: Config,
) -> HeteroMLP:
    model = HeteroMLP(
        cfg.input_dim,
        cfg.hidden_dim,
        cfg.feature_dim,
        cfg.variance_floor,
    ).to(DEVICE)

    x_t = to_tensor(x)
    y_t = to_tensor(y)

    fit_idx, val_idx = train_validation_split(
        len(x),
        cfg.validation_fraction,
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
# Last-layer geometry
# ============================================================

def add_bias_column(phi: np.ndarray) -> np.ndarray:
    ones = np.ones((phi.shape[0], 1), dtype=np.float64)
    return np.concatenate([phi.astype(np.float64), ones], axis=1)


def invert_precision(precision: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(precision)


def compute_weighted_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    cfg: Config,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train)
    weights = 1.0 / np.maximum(
        h2_train.reshape(-1).astype(np.float64),
        cfg.variance_floor,
    )

    precision = prior_precision * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ (h_aug * weights[:, None])
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)

    return invert_precision(precision)


def compute_unweighted_covariance(
    phi_train: np.ndarray,
    prior_precision: float,
    cfg: Config,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train)

    precision = prior_precision * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ h_aug
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)

    return invert_precision(precision)


def quadratic_form(
    phi: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    h_aug = add_bias_column(phi)
    values = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(values, 0.0).astype(np.float32)


def choose_prior_precision(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
    weighted: bool,
) -> float:
    # Prior precision is selected using training data only.
    fit_idx, val_idx = train_validation_split(
        len(data["x_train"]),
        cfg.validation_fraction,
    )

    _, h2_fit, phi_fit = predict_hetero(model, data["x_train"][fit_idx])
    mu_val, h2_val, phi_val = predict_hetero(model, data["x_train"][val_idx])
    y_val = data["y_train_std"][val_idx]

    best_lambda = float(cfg.lambda_grid[0])
    best_nll = float("inf")

    for lam in cfg.lambda_grid:
        if weighted:
            covariance = compute_weighted_covariance(
                phi_fit,
                h2_fit,
                float(lam),
                cfg,
            )
        else:
            covariance = compute_unweighted_covariance(
                phi_fit,
                float(lam),
                cfg,
            )

        epi_val = quadratic_form(phi_val, covariance)
        total_var = np.maximum(h2_val + epi_val, cfg.variance_floor)
        residual2 = (y_val - mu_val) ** 2
        nll = 0.5 * np.mean(residual2 / total_var + np.log(total_var))

        if nll < best_nll:
            best_nll = float(nll)
            best_lambda = float(lam)

    return best_lambda


# ============================================================
# Four conformal methods
# ============================================================

def interval_from_scale(
    mu_cal: np.ndarray,
    scale_cal: np.ndarray,
    mu_test: np.ndarray,
    scale_test: np.ndarray,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scale_cal = np.maximum(scale_cal, 1e-8)
    scale_test = np.maximum(scale_test, 1e-8)

    scores = np.abs(data["y_cal_std"] - mu_cal) / scale_cal
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_std = mu_test - qhat * scale_test
    upper_std = mu_test + qhat * scale_test

    lower = unstandardize_y(lower_std, data["y_mean"], data["y_scale"])
    upper = unstandardize_y(upper_std, data["y_mean"], data["y_scale"])

    return lower, upper, float(qhat)


def run_lacp(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    mu_cal, h2_cal, _ = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, _ = predict_hetero(model, data["x_test"])

    lower, upper, qhat = interval_from_scale(
        mu_cal,
        np.sqrt(np.maximum(h2_cal, cfg.variance_floor)),
        mu_test,
        np.sqrt(np.maximum(h2_test, cfg.variance_floor)),
        data,
        cfg,
    )

    return lower, upper, {"qhat": qhat, "epistemic_fraction": np.nan}


def run_unweighted_laplace(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    chosen_lambda = choose_prior_precision(model, data, cfg, weighted=False)

    _, _, phi_train = predict_hetero(model, data["x_train"])
    mu_cal, h2_cal, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, phi_test = predict_hetero(model, data["x_test"])

    covariance = compute_unweighted_covariance(
        phi_train,
        chosen_lambda,
        cfg,
    )
    epi_cal = quadratic_form(phi_cal, covariance)
    epi_test = quadratic_form(phi_test, covariance)

    total_var_cal = np.maximum(h2_cal + epi_cal, cfg.variance_floor)
    total_var_test = np.maximum(h2_test + epi_test, cfg.variance_floor)

    lower, upper, qhat = interval_from_scale(
        mu_cal,
        np.sqrt(total_var_cal),
        mu_test,
        np.sqrt(total_var_test),
        data,
        cfg,
    )

    return lower, upper, {"qhat": qhat, "epistemic_fraction": np.nan}


def run_lwcp(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    # Ordinary ridge leverage in the same learned representation.
    _, _, phi_train = predict_hetero(model, data["x_train"])
    mu_cal, _, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, _, phi_test = predict_hetero(model, data["x_test"])

    covariance = compute_unweighted_covariance(
        phi_train,
        cfg.lwcp_ridge,
        cfg,
    )
    leverage_cal = quadratic_form(phi_cal, covariance)
    leverage_test = quadratic_form(phi_test, covariance)

    # Equivalent to score = |residual| / sqrt(1 + leverage).
    scale_cal = np.sqrt(1.0 + leverage_cal)
    scale_test = np.sqrt(1.0 + leverage_test)

    lower, upper, qhat = interval_from_scale(
        mu_cal,
        scale_cal,
        mu_test,
        scale_test,
        data,
        cfg,
    )

    return lower, upper, {"qhat": qhat, "epistemic_fraction": np.nan}


def run_claps(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    chosen_lambda = choose_prior_precision(model, data, cfg, weighted=True)

    _, h2_train, phi_train = predict_hetero(model, data["x_train"])
    mu_cal, h2_cal, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, phi_test = predict_hetero(model, data["x_test"])

    covariance = compute_weighted_covariance(
        phi_train,
        h2_train,
        chosen_lambda,
        cfg,
    )
    epi_cal = quadratic_form(phi_cal, covariance)
    epi_test = quadratic_form(phi_test, covariance)

    total_var_cal = np.maximum(h2_cal + epi_cal, cfg.variance_floor)
    total_var_test = np.maximum(h2_test + epi_test, cfg.variance_floor)

    lower, upper, qhat = interval_from_scale(
        mu_cal,
        np.sqrt(total_var_cal),
        mu_test,
        np.sqrt(total_var_test),
        data,
        cfg,
    )

    epistemic_fraction = float(
        np.mean(epi_test / np.maximum(total_var_test, cfg.variance_floor))
    )

    return lower, upper, {
        "qhat": qhat,
        "epistemic_fraction": epistemic_fraction,
    }


# ============================================================
# Four reported quantities
# ============================================================

def compute_reported_quantities(
    method_name: str,
    seed: int,
    y_test: np.ndarray,
    region_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    epistemic_fraction: float,
    cfg: Config,
) -> Dict[str, float]:
    y = y_test.reshape(-1)
    lower = lower.reshape(-1)
    upper = upper.reshape(-1)

    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = interval_score(y, lower, upper, cfg.alpha)

    dense_mask = region_test == REGION_DENSE_LOW
    sparse_mask = region_test == REGION_SPARSE_LOW

    dense_width = float(np.mean(width[dense_mask]))
    sparse_width = float(np.mean(width[sparse_mask]))

    return {
        "method": method_name,
        "seed": seed,
        "marginal_coverage": float(np.mean(covered)),
        "sparse_dense_width_ratio": (
            sparse_width / max(dense_width, 1e-12)
        ),
        "sparse_interval_score": float(np.mean(score[sparse_mask])),
        "claps_epistemic_fraction": (
            float(epistemic_fraction)
            if method_name == "CLAPS"
            else np.nan
        ),
    }


# ============================================================
# One setting and the two sweeps
# ============================================================

def run_single_setting(
    n_train: int,
    sparse_retention: float,
    seed: int,
    cfg: Config,
) -> List[Dict[str, float]]:
    data = make_dataset(
        n_train=n_train,
        sparse_retention=sparse_retention,
        seed=seed,
        cfg=cfg,
    )

    hetero_model = train_hetero_model(
        data["x_train"],
        data["y_train_std"],
        cfg,
    )

    method_runners = [
        ("LACP", lambda: run_lacp(hetero_model, data, cfg)),
        (
            "Unweighted Laplace",
            lambda: run_unweighted_laplace(hetero_model, data, cfg),
        ),
        ("LWCP", lambda: run_lwcp(hetero_model, data, cfg)),
        ("CLAPS", lambda: run_claps(hetero_model, data, cfg)),
    ]

    rows: List[Dict[str, float]] = []

    for method_name, runner in method_runners:
        lower, upper, diagnostics = runner()
        row = compute_reported_quantities(
            method_name=method_name,
            seed=seed,
            y_test=data["y_test"],
            region_test=data["region_test"],
            lower=lower,
            upper=upper,
            epistemic_fraction=diagnostics["epistemic_fraction"],
            cfg=cfg,
        )
        row["n_train"] = int(n_train)
        row["sparse_retention"] = float(sparse_retention)
        rows.append(row)

    del hetero_model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def run_experiment(cfg: Config) -> pd.DataFrame:
    all_rows: List[Dict[str, float]] = []

    # The setting shared by the two sweeps is computed only once per seed.
    cache: Dict[Tuple[int, float, int], List[Dict[str, float]]] = {}

    size_settings = [
        ("training_size", int(n), float(cfg.fixed_sparse_retention))
        for n in cfg.n_train_grid
    ]
    support_settings = [
        ("support", int(cfg.fixed_n_train), float(retention))
        for retention in cfg.sparse_retention_grid
    ]
    jobs = size_settings + support_settings

    total_jobs = len(jobs) * cfg.num_seeds
    completed = 0
    start_time = time.time()

    for panel, n_train, retention in jobs:
        for seed in range(cfg.num_seeds):
            completed += 1
            key = (n_train, round(retention, 8), seed)

            if cfg.verbose:
                elapsed = time.time() - start_time
                cache_label = "cached" if key in cache else "fit"
                print(
                    f"[{completed}/{total_jobs}] panel={panel} | "
                    f"n_train={n_train} | retention={retention:g} | "
                    f"seed={seed} | {cache_label} | elapsed={elapsed:.1f}s"
                )

            if key not in cache:
                cache[key] = run_single_setting(
                    n_train=n_train,
                    sparse_retention=retention,
                    seed=seed,
                    cfg=cfg,
                )

            for cached_row in cache[key]:
                row = dict(cached_row)
                row["panel"] = panel
                row["setting_value"] = (
                    float(n_train)
                    if panel == "training_size"
                    else float(retention)
                )
                all_rows.append(row)

    return pd.DataFrame(all_rows)


# ============================================================
# Compact paper-style tables
# ============================================================

def build_panel_table(
    results: pd.DataFrame,
    panel: str,
    cfg: Config,
) -> pd.DataFrame:
    panel_df = results[results["panel"] == panel].copy()
    rows = []

    setting_values = sorted(panel_df["setting_value"].unique())

    for setting_value in setting_values:
        setting_df = panel_df[panel_df["setting_value"] == setting_value]

        for method_name in METHOD_ORDER:
            method_df = setting_df[setting_df["method"] == method_name]
            if method_df.empty:
                continue

            row = {
                "Train n" if panel == "training_size" else "Sparse retention": (
                    int(setting_value)
                    if panel == "training_size"
                    else f"{setting_value:g}"
                ),
                "Method": method_name,
            }

            metric_specs = [
                ("marginal_coverage", "Marg. Cov."),
                ("sparse_dense_width_ratio", "Sparse/Dense Width"),
                ("sparse_interval_score", "Sparse Int. Score"),
                ("claps_epistemic_fraction", "CLAPS Epi. Frac."),
            ]

            for metric, label in metric_specs:
                values = method_df[metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    row[label] = "—"
                    continue

                mean_value = float(np.mean(values))
                se_value = (
                    float(np.std(values, ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
                row[label] = format_mean_se(
                    mean_value,
                    se_value,
                    cfg.summary_digits,
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    results_df = run_experiment(CFG)

    training_size_table = build_panel_table(
        results_df,
        panel="training_size",
        cfg=CFG,
    )
    support_table = build_panel_table(
        results_df,
        panel="support",
        cfg=CFG,
    )

    print("\nFinished Experiment 3.")
    print("\nA. Training-size sweep: mean ± standard error over seeds")
    display(training_size_table)

    print("\nB. Local-support sweep: mean ± standard error over seeds")
    display(support_table)

    if CFG.display_raw_results:
        print("\nRaw per-seed results")
        display(results_df.round(CFG.summary_digits))

# ============================================================
# Experiment 3: compact main-text figure
#
# Run this cell after Experiment 3 so that `results_df` exists.
#
# Layout:
#   (a) Width allocation vs. training size
#   (b) Width allocation vs. local support
#   (c) Sparse-region interval score vs. training size
#   (d) Sparse-region interval score vs. local support
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

METHOD_ORDER = [
    "LACP",
    "Unweighted Laplace",
    "LWCP",
    "CLAPS",
]

METHOD_STYLES = {
    "LACP": {
        "marker": "o",
        "linestyle": "--",
        "color": "tab:gray",
    },
    "Unweighted Laplace": {
        "marker": "s",
        "linestyle": "-.",
        "color": "tab:orange",
    },
    "LWCP": {
        "marker": "^",
        "linestyle": ":",
        "color": "tab:green",
    },
    "CLAPS": {
        "marker": "D",
        "linestyle": "-",
        "color": "tab:blue",
    },
}

EPISTEMIC_COLOR = "tab:red"

TRAIN_SIZE_ORDER = [100, 250, 500, 1000, 2000]

SUPPORT_ORDER = [1.00, 0.20, 0.10, 0.05, 0.025, 0.01]

BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260801


# ------------------------------------------------------------
# Bootstrap mean and confidence interval
# ------------------------------------------------------------

def bootstrap_mean_ci(
    values,
    repeats=BOOTSTRAP_REPEATS,
    seed=BOOTSTRAP_SEED,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

    if len(values) == 1:
        return mean_value, mean_value, mean_value

    rng = np.random.default_rng(seed)

    sampled_indices = rng.integers(
        low=0,
        high=len(values),
        size=(repeats, len(values)),
    )
    bootstrap_means = values[sampled_indices].mean(axis=1)

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return mean_value, float(ci_low), float(ci_high)


def summarize_metric(
    panel_df,
    method,
    setting_order,
    metric,
    seed_offset=0,
):
    means = []
    lower_errors = []
    upper_errors = []

    method_df = panel_df[panel_df["method"] == method]

    for index, setting in enumerate(setting_order):
        setting_mask = np.isclose(
            method_df["setting_value"].astype(float),
            float(setting),
        )

        values = (
            method_df.loc[setting_mask, metric]
            .dropna()
            .to_numpy(dtype=float)
        )

        mean_value, ci_low, ci_high = bootstrap_mean_ci(
            values,
            seed=BOOTSTRAP_SEED + seed_offset + index,
        )

        means.append(mean_value)
        lower_errors.append(mean_value - ci_low)
        upper_errors.append(ci_high - mean_value)

    return (
        np.asarray(means),
        np.asarray([lower_errors, upper_errors]),
    )


# ------------------------------------------------------------
# Plot one sweep
# ------------------------------------------------------------

def plot_sweep(
    ax_width,
    ax_score,
    panel_df,
    setting_order,
    setting_labels,
    x_label,
    panel_letters,
    show_left_axis,
    show_right_epistemic_axis,
):
    x_positions = np.arange(len(setting_order))

    method_handles = {}

    # --------------------------------------------------------
    # Sparse/dense width ratio
    # --------------------------------------------------------

    for method_index, method in enumerate(METHOD_ORDER):
        means, errors = summarize_metric(
            panel_df=panel_df,
            method=method,
            setting_order=setting_order,
            metric="sparse_dense_width_ratio",
            seed_offset=1000 * method_index,
        )

        style = METHOD_STYLES[method]

        errorbar = ax_width.errorbar(
            x_positions,
            means,
            yerr=errors,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.7,
            markersize=5,
            capsize=2.5,
            label=method,
        )

        method_handles[method] = errorbar.lines[0]

    ax_width.axhline(
        1.0,
        linewidth=1.0,
        linestyle=":",
        color="black",
        alpha=0.6,
    )

    ax_width.set_xticks(x_positions)
    ax_width.set_xticklabels(setting_labels)
    ax_width.grid(axis="y", alpha=0.25)

    if show_left_axis:
        ax_width.set_ylabel("Sparse/dense width ratio")
    else:
        ax_width.set_ylabel("")
        ax_width.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )
        #ax_width.spines["left"].set_visible(False)

    # --------------------------------------------------------
    # CLAPS epistemic fraction on secondary axis
    # --------------------------------------------------------

    ax_epi = ax_width.twinx()

    epi_means, epi_errors = summarize_metric(
        panel_df=panel_df,
        method="CLAPS",
        setting_order=setting_order,
        metric="claps_epistemic_fraction",
        seed_offset=9000,
    )

    epi_means = 100.0 * epi_means
    epi_errors = 100.0 * epi_errors

    epi_errorbar = ax_epi.errorbar(
        x_positions,
        epi_means,
        yerr=epi_errors,
        marker="X",
        linestyle="--",
        linewidth=1.5,
        markersize=5.5,
        capsize=2.5,
        color=EPISTEMIC_COLOR,
        label="CLAPS epistemic fraction",
    )

    if show_right_epistemic_axis:
        ax_epi.set_ylabel("CLAPS epistemic fraction (%)")
    else:
        ax_epi.set_ylabel("")
        ax_epi.tick_params(
            axis="y",
            which="both",
            right=False,
            labelright=False,
        )

    # --------------------------------------------------------
    # Sparse-region interval score
    # --------------------------------------------------------

    for method_index, method in enumerate(METHOD_ORDER):
        means, errors = summarize_metric(
            panel_df=panel_df,
            method=method,
            setting_order=setting_order,
            metric="sparse_interval_score",
            seed_offset=20_000 + 1000 * method_index,
        )

        style = METHOD_STYLES[method]

        ax_score.errorbar(
            x_positions,
            means,
            yerr=errors,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.7,
            markersize=5,
            capsize=2.5,
            label=method,
        )

    ax_score.set_xlabel(x_label)
    ax_score.set_xticks(x_positions)
    ax_score.set_xticklabels(setting_labels)
    ax_score.grid(axis="y", alpha=0.25)

    if show_left_axis:
        ax_score.set_ylabel("Sparse-region interval score")
    else:
        ax_score.set_ylabel("")
        ax_score.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

    # --------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------

    ax_width.text(
        0.02,
        0.96,
        panel_letters[0],
        transform=ax_width.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    ax_score.text(
        0.02,
        0.96,
        panel_letters[1],
        transform=ax_score.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )

    return method_handles, epi_errorbar.lines[0], ax_epi


# ------------------------------------------------------------
# Construct figure
# ------------------------------------------------------------

if "results_df" not in globals():
    raise RuntimeError(
        "results_df was not found. Run the Experiment 3 cell first."
    )

training_df = results_df[
    results_df["panel"] == "training_size"
].copy()

support_df = results_df[
    results_df["panel"] == "support"
].copy()

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(7.4, 5.2),
    sharex="col",
    sharey="row",
    constrained_layout=False,
)

training_handles, training_epi_handle, training_epi_ax = plot_sweep(
    ax_width=axes[0, 0],
    ax_score=axes[1, 0],
    panel_df=training_df,
    setting_order=TRAIN_SIZE_ORDER,
    setting_labels=[str(value) for value in TRAIN_SIZE_ORDER],
    x_label="Training-set size",
    panel_letters=("(a)", "(c)"),
    show_left_axis=True,
    show_right_epistemic_axis=False,
)

support_handles, support_epi_handle, support_epi_ax = plot_sweep(
    ax_width=axes[0, 1],
    ax_score=axes[1, 1],
    panel_df=support_df,
    setting_order=SUPPORT_ORDER,
    setting_labels=[
        "1.00",
        "0.20",
        "0.10",
        "0.05",
        "0.025",
        "0.01",
    ],
    x_label="Sparse-region retention",
    panel_letters=("(b)", "(d)"),
    show_left_axis=False,
    show_right_epistemic_axis=True,
)

all_epi_values = (
    results_df["claps_epistemic_fraction"]
    .dropna()
    .to_numpy(dtype=float)
    * 100.0
)

epi_upper = max(
    1.0,
    np.ceil(np.max(all_epi_values) * 1.25),
)

training_epi_ax.set_ylim(0.0, epi_upper)
support_epi_ax.set_ylim(0.0, epi_upper)

# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------

legend_handles = [
    training_handles[method]
    for method in METHOD_ORDER
]
legend_labels = METHOD_ORDER.copy()

legend_handles.append(training_epi_handle)
legend_labels.append("CLAPS epistemic fraction")

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=5,
    frameon=False,
)

fig.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.95]
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fig.savefig(
    "experiment3_regime_study.pdf",
    bbox_inches="tight",
)

fig.savefig(
    "experiment3_regime_study.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


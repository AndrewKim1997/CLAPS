
# ============================================================
# Experiment 2: Isolating the Noise-Weighted Feature Geometry
# Single-cell Google Colab implementation
#
# Purpose:
#   Separate the effect of adding last-layer epistemic uncertainty
#   from the effect of heteroscedastic precision weighting.
#
# Methods:
#   1) LACP
#   2) Unweighted Laplace
#   3) LWCP
#   4) CLAPS
#
# Four regions:
#   dense-low-noise, sparse-low-noise,
#   dense-high-noise, sparse-high-noise
#
# Only four evaluation metrics are reported:
#   1) Marginal coverage
#   2) Sparse/dense width ratio under low noise
#   3) Sparse/dense width ratio under high noise
#   4) Overall interval score
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
    # Five seeds are used initially so the first Colab run is manageable.
    num_seeds: int = 30

    # Data sizes
    input_dim: int = 10
    n_train: int = 1200
    n_cal: int = 2500
    n_test: int = 12000

    # Evaluation distribution: four equally represented regions.
    target_region_probs: Tuple[float, float, float, float] = (
        0.25, 0.25, 0.25, 0.25
    )

    # Training distribution: both sparse regions receive the same small mass.
    # This makes the sparse-low and sparse-high regions geometrically comparable.
    train_region_probs: Tuple[float, float, float, float] = (
        0.46, 0.04, 0.46, 0.04
    )

    # Data-generating process
    cluster_std: float = 0.70
    low_noise_std: float = 0.12
    high_noise_std: float = 0.70
    sparse_bump_amplitude: float = 0.90
    sparse_bump_bandwidth: float = 0.85

    # Nominal coverage
    alpha: float = 0.10

    # Shared heteroscedastic neural model
    hidden_dim: int = 64
    feature_dim: int = 64
    max_epochs: int = 250
    patience: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5
    variance_floor: float = 1e-4

    # Last-layer posterior selection
    lambda_grid: Tuple[float, ...] = (
        1e-5, 1e-4, 1e-3, 1e-2,
        1e-1, 1.0, 10.0, 100.0
    )
    jitter: float = 1e-6

    # LWCP ridge
    lwcp_ridge: float = 1e-3

    # Paired uncertainty summary
    bootstrap_repeats: int = 5000

    # Display
    verbose: bool = True
    summary_digits: int = 4
    display_raw_results: bool = False


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGION_DENSE_LOW = 0
REGION_SPARSE_LOW = 1
REGION_DENSE_HIGH = 2
REGION_SPARSE_HIGH = 3

print(f"Using device: {DEVICE}")
print("Experiment 2: noise-weighted feature-geometry isolation")
print(
    "Reported metrics: Marginal Cov., Low-noise Sparse/Dense Width, "
    "High-noise Sparse/Dense Width, Int. Score"
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
    return idx[n_val:], idx[:n_val]


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


# ============================================================
# Four-region synthetic data
# ============================================================

def region_centers(input_dim: int) -> np.ndarray:
    centers = np.zeros((4, input_dim), dtype=np.float32)

    # A symmetric 2 x 2 layout in the first two coordinates.
    # Left column: dense support. Right column: sparse support.
    # Bottom row: low noise. Top row: high noise.
    centers[REGION_DENSE_LOW, 0:2] = np.array([-2.5, -2.5])
    centers[REGION_SPARSE_LOW, 0:2] = np.array([2.5, -2.5])
    centers[REGION_DENSE_HIGH, 0:2] = np.array([-2.5, 2.5])
    centers[REGION_SPARSE_HIGH, 0:2] = np.array([2.5, 2.5])

    return centers


def sample_region_mixture(
    n: int,
    probs: Tuple[float, float, float, float],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    probs_array = np.asarray(probs, dtype=float)
    probs_array = probs_array / probs_array.sum()

    region = np.random.choice(4, size=n, p=probs_array)
    centers = region_centers(cfg.input_dim)
    x = centers[region] + cfg.cluster_std * np.random.randn(n, cfg.input_dim)

    return x.astype(np.float32), region.astype(np.int64)


def true_mean(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    base = (
        0.75 * np.sin(x[:, 0:1])
        + 0.30 * x[:, 1:2]
        + 0.18 * np.square(x[:, 2:3])
        - 0.15 * x[:, 0:1] * x[:, 1:2]
        + 0.12 * np.tanh(x[:, 3:4])
    )

    centers = region_centers(cfg.input_dim)
    sparse_low_center = centers[REGION_SPARSE_LOW, 0:2]
    sparse_high_center = centers[REGION_SPARSE_HIGH, 0:2]

    distance2_low = np.sum(
        np.square(x[:, 0:2] - sparse_low_center[None, :]),
        axis=1,
        keepdims=True,
    )
    distance2_high = np.sum(
        np.square(x[:, 0:2] - sparse_high_center[None, :]),
        axis=1,
        keepdims=True,
    )

    # The same local mean complexity is placed in both sparse regions.
    # Their main difference is observation noise, not function geometry.
    local_bumps = cfg.sparse_bump_amplitude * (
        np.exp(-distance2_low / (2.0 * cfg.sparse_bump_bandwidth ** 2))
        + np.exp(-distance2_high / (2.0 * cfg.sparse_bump_bandwidth ** 2))
    )

    return (base + local_bumps).astype(np.float32)


def true_noise_std(region: np.ndarray, cfg: Config) -> np.ndarray:
    sigma = np.full((len(region), 1), cfg.low_noise_std, dtype=np.float32)
    high_mask = (
        (region == REGION_DENSE_HIGH)
        | (region == REGION_SPARSE_HIGH)
    )
    sigma[high_mask] = cfg.high_noise_std
    return sigma


def generate_y(
    x: np.ndarray,
    region: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = true_mean(x, cfg)
    sigma = true_noise_std(region, cfg)
    y = mean + sigma * np.random.randn(len(x), 1)
    return y.astype(np.float32), mean, sigma


def make_dataset(cfg: Config) -> Dict[str, np.ndarray]:
    x_train_raw, region_train = sample_region_mixture(
        cfg.n_train,
        cfg.train_region_probs,
        cfg,
    )
    x_cal_raw, region_cal = sample_region_mixture(
        cfg.n_cal,
        cfg.target_region_probs,
        cfg,
    )
    x_test_raw, region_test = sample_region_mixture(
        cfg.n_test,
        cfg.target_region_probs,
        cfg,
    )

    y_train, f_train, sigma_train = generate_y(
        x_train_raw,
        region_train,
        cfg,
    )
    y_cal, f_cal, sigma_cal = generate_y(
        x_cal_raw,
        region_cal,
        cfg,
    )
    y_test, f_test, sigma_test = generate_y(
        x_test_raw,
        region_test,
        cfg,
    )

    # Standardization uses training data only.
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
        "y_test_std": ((y_test - y_mean) / y_scale).astype(np.float32),
        "region_train": region_train,
        "region_cal": region_cal,
        "region_test": region_test,
        "f_train": f_train,
        "f_cal": f_cal,
        "f_test": f_test,
        "sigma_train": sigma_train,
        "sigma_cal": sigma_cal,
        "sigma_test": sigma_test,
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

    train_idx, val_idx = train_validation_split(
        len(x),
        cfg.validation_fraction,
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
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

    return np.linalg.inv(precision)


def compute_unweighted_laplace_covariance(
    phi_train: np.ndarray,
    prior_precision: float,
    cfg: Config,
) -> np.ndarray:
    h_aug = add_bias_column(phi_train)

    precision = prior_precision * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ h_aug
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)

    return np.linalg.inv(precision)


def quadratic_form(
    phi: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    h_aug = add_bias_column(phi)
    values = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(values, 0.0).astype(np.float32)


def choose_prior_precision(
    model: HeteroMLP,
    x_train: np.ndarray,
    y_train_std: np.ndarray,
    cfg: Config,
    weighted: bool,
) -> float:
    # Lambda is selected using training data only.
    fit_idx, val_idx = train_validation_split(
        len(x_train),
        cfg.validation_fraction,
    )

    _, h2_fit, phi_fit = predict_hetero(model, x_train[fit_idx])
    mu_val, h2_val, phi_val = predict_hetero(model, x_train[val_idx])
    y_val = y_train_std[val_idx]

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
            covariance = compute_unweighted_laplace_covariance(
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

def run_lacp(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_cal, h2_cal, _ = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, _ = predict_hetero(model, data["x_test"])

    h_cal = np.sqrt(np.maximum(h2_cal, cfg.variance_floor))
    h_test = np.sqrt(np.maximum(h2_test, cfg.variance_floor))

    scores = np.abs(data["y_cal_std"] - mu_cal) / h_cal
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_std = mu_test - qhat * h_test
    upper_std = mu_test + qhat * h_test

    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def run_unweighted_laplace(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    chosen_lambda = choose_prior_precision(
        model,
        data["x_train"],
        data["y_train_std"],
        cfg,
        weighted=False,
    )

    _, _, phi_train = predict_hetero(model, data["x_train"])
    mu_cal, h2_cal, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, h2_test, phi_test = predict_hetero(model, data["x_test"])

    covariance = compute_unweighted_laplace_covariance(
        phi_train,
        chosen_lambda,
        cfg,
    )
    epi_cal = quadratic_form(phi_cal, covariance)
    epi_test = quadratic_form(phi_test, covariance)

    total_var_cal = np.maximum(h2_cal + epi_cal, cfg.variance_floor)
    total_var_test = np.maximum(h2_test + epi_test, cfg.variance_floor)

    scores = np.abs(data["y_cal_std"] - mu_cal) / np.sqrt(total_var_cal)
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_std = mu_test - qhat * np.sqrt(total_var_test)
    upper_std = mu_test + qhat * np.sqrt(total_var_test)

    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def run_lwcp(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    # Leverage-only local adaptation using the same learned representation.
    _, _, phi_train = predict_hetero(model, data["x_train"])
    mu_cal, _, phi_cal = predict_hetero(model, data["x_cal"])
    mu_test, _, phi_test = predict_hetero(model, data["x_test"])

    covariance = compute_unweighted_laplace_covariance(
        phi_train,
        cfg.lwcp_ridge,
        cfg,
    )
    leverage_cal = quadratic_form(phi_cal, covariance)
    leverage_test = quadratic_form(phi_test, covariance)

    weight_cal = 1.0 / np.sqrt(1.0 + leverage_cal)
    weight_test = 1.0 / np.sqrt(1.0 + leverage_test)

    scores = np.abs(data["y_cal_std"] - mu_cal) * weight_cal
    qhat = finite_sample_quantile(scores, cfg.alpha)

    half_width_test = qhat / np.maximum(weight_test, 1e-8)
    lower_std = mu_test - half_width_test
    upper_std = mu_test + half_width_test

    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


def run_claps(
    model: HeteroMLP,
    data: Dict[str, np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    chosen_lambda = choose_prior_precision(
        model,
        data["x_train"],
        data["y_train_std"],
        cfg,
        weighted=True,
    )

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

    scores = np.abs(data["y_cal_std"] - mu_cal) / np.sqrt(total_var_cal)
    qhat = finite_sample_quantile(scores, cfg.alpha)

    lower_std = mu_test - qhat * np.sqrt(total_var_test)
    upper_std = mu_test + qhat * np.sqrt(total_var_test)

    return (
        unstandardize_y(lower_std, data["y_mean"], data["y_scale"]),
        unstandardize_y(upper_std, data["y_mean"], data["y_scale"]),
    )


# ============================================================
# Four evaluation metrics only
# ============================================================

def compute_four_metrics(
    method_name: str,
    seed: int,
    y_test: np.ndarray,
    region_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    y = y_test.reshape(-1)
    lower = lower.reshape(-1)
    upper = upper.reshape(-1)

    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = interval_score(y, lower, upper, cfg.alpha)

    dense_low_mask = region_test == REGION_DENSE_LOW
    sparse_low_mask = region_test == REGION_SPARSE_LOW
    dense_high_mask = region_test == REGION_DENSE_HIGH
    sparse_high_mask = region_test == REGION_SPARSE_HIGH

    dense_low_width = float(np.mean(width[dense_low_mask]))
    sparse_low_width = float(np.mean(width[sparse_low_mask]))
    dense_high_width = float(np.mean(width[dense_high_mask]))
    sparse_high_width = float(np.mean(width[sparse_high_mask]))

    return {
        "method": method_name,
        "seed": seed,
        "marginal_coverage": float(np.mean(covered)),
        "low_noise_support_ratio": (
            sparse_low_width / max(dense_low_width, 1e-12)
        ),
        "high_noise_support_ratio": (
            sparse_high_width / max(dense_high_width, 1e-12)
        ),
        "interval_score": float(np.mean(score)),
    }


# ============================================================
# Experiment loop
# ============================================================

def run_single_seed(seed: int, cfg: Config) -> List[Dict[str, float]]:
    set_seed(seed)
    data = make_dataset(cfg)

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

    results = []
    for method_name, runner in method_runners:
        lower, upper = runner()
        results.append(
            compute_four_metrics(
                method_name,
                seed,
                data["y_test"],
                data["region_test"],
                lower,
                upper,
                cfg,
            )
        )

    del hetero_model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_experiment(cfg: Config) -> pd.DataFrame:
    all_results = []
    start_time = time.time()

    for seed in range(cfg.num_seeds):
        if cfg.verbose:
            elapsed = time.time() - start_time
            print(
                f"[{seed + 1}/{cfg.num_seeds}] "
                f"seed={seed} | elapsed={elapsed:.1f}s"
            )

        all_results.extend(run_single_seed(seed, cfg))

    return pd.DataFrame(all_results)


# ============================================================
# Summaries using the same four metrics
# ============================================================

METHOD_ORDER = [
    "LACP",
    "Unweighted Laplace",
    "LWCP",
    "CLAPS",
]

METRICS = [
    ("marginal_coverage", "Marg. Cov."),
    ("low_noise_support_ratio", "Low-noise Sparse/Dense Width"),
    ("high_noise_support_ratio", "High-noise Sparse/Dense Width"),
    ("interval_score", "Int. Score"),
]


def format_mean_se(mean_value: float, se_value: float, digits: int) -> str:
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}"


def build_summary_table(results: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        method_df = results[results["method"] == method]
        if method_df.empty:
            continue

        row = {"Method": method}
        for metric, label in METRICS:
            values = method_df[metric].to_numpy(dtype=float)
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


def paired_bootstrap_claps_comparisons(
    results: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    claps = (
        results[results["method"] == "CLAPS"]
        .set_index("seed")
        .sort_index()
    )

    comparators = ["LACP", "Unweighted Laplace", "LWCP"]
    rng = np.random.default_rng(20260731)
    rows = []

    for comparator_name in comparators:
        comparator = (
            results[results["method"] == comparator_name]
            .set_index("seed")
            .sort_index()
        )
        common_seeds = claps.index.intersection(comparator.index)

        for metric, label in METRICS:
            differences = (
                claps.loc[common_seeds, metric].to_numpy(dtype=float)
                - comparator.loc[common_seeds, metric].to_numpy(dtype=float)
            )

            if len(differences) == 1:
                ci_low = ci_high = float(differences[0])
            else:
                sampled_indices = rng.integers(
                    0,
                    len(differences),
                    size=(cfg.bootstrap_repeats, len(differences)),
                )
                bootstrap_means = differences[sampled_indices].mean(axis=1)
                ci_low, ci_high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )

            rows.append({
                "Comparison": f"CLAPS - {comparator_name}",
                "Metric": label,
                "Mean difference": float(np.mean(differences)),
                "95% CI low": float(ci_low),
                "95% CI high": float(ci_high),
            })

    return pd.DataFrame(rows)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    results_df = run_experiment(CFG)
    summary_table = build_summary_table(results_df, CFG)
    paired_table = paired_bootstrap_claps_comparisons(results_df, CFG)

    print("\nFinished Experiment 2.")
    print("\nMain summary: mean ± standard error over seeds")
    display(summary_table)

    print("\nPaired comparisons against CLAPS")
    display(paired_table.round(CFG.summary_digits))

    if CFG.display_raw_results:
        print("\nRaw per-seed results")
        display(results_df.round(CFG.summary_digits))


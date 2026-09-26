
# -*- coding: utf-8 -*-
"""
Appendix Experiment: OOD Stress Test under a Fixed Representation
=================================================================

Standalone Google Colab / Python implementation aligned with Experiment 1
in the CLAPS revision code.

Purpose
-------
This experiment probes whether the last-layer correction reacts to input
shift only when that shift remains visible in the frozen learned
representation. It uses the same 10-dimensional weak-support synthetic
problem, heteroscedastic neural regressor, training-only prior selection,
and split-conformal calibration protocol as Experiment 1.

Test conditions
---------------
1. ID reference
   Samples from the original calibration/test target mixture.

2. Feature-visible OOD
   ID anchors are shifted in the four signal coordinates. Candidate points
   must lie beyond the ID 99th percentile in both raw-input and learned-
   feature kNN distance.

3. Feature-collapsed OOD
   Candidate inputs are optimized to change the true conditional mean while
   remaining close to an ID anchor in the frozen learned representation.
   They must lie beyond the ID 99th percentile in raw-input kNN distance,
   remain below the ID 90th percentile in learned-feature kNN distance, and
   change the true conditional mean by at least a prespecified fraction of
   the training-response scale. This is a deliberately harmful representation-
   collapse stress test rather than a target-irrelevant nuisance shift.

Methods
-------
LACP, Unweighted Laplace, and CLAPS share one fitted heteroscedastic neural
regressor. Their conformal quantiles are calibrated once on the unchanged
ID calibration sample and then applied without refitting to all three test
conditions.

Important interpretation
------------------------
The OOD samples are not exchangeable with the calibration sample. Their
empirical coverage is therefore a descriptive stress-test outcome rather
than a consequence of the split-conformal finite-sample guarantee.

Created outputs
---------------
condition_diagnostics_df
    Per-seed raw/feature distances, point-prediction error, epistemic
    fractions, and width ratios for each test condition.

method_results_df
    Per-seed coverage, mean width, and interval score for each method and
    condition.

diagnostic_summary_df
    Mean ± standard error summary of the condition diagnostics.

method_summary_df
    Mean ± standard error summary of method performance by condition.

paired_effects_df
    Paired bootstrap CLAPS-versus-LACP effects by condition.

figure_points_df
    Per-point diagnostics for one representative seed, used to create the
    raw-distance versus feature-distance scatter plot.

Saved files
-----------
ood_stress_condition_diagnostics.csv
ood_stress_method_results.csv
ood_stress_diagnostic_summary.csv
ood_stress_method_summary.csv
ood_stress_paired_effects.csv
ood_stress_scatter_seed0.csv
ood_stress_fixed_representation.pdf
ood_stress_fixed_representation.png
"""

from __future__ import annotations

import gc
import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class OODConfig:
    # Final paper run.
    num_seeds: int = 30

    # Same core sample sizes as Experiment 1.
    input_dim: int = 10
    n_train: int = 1000
    n_cal: int = 2000
    n_id_test: int = 5000
    n_ood_test: int = 500

    # Calibration/test target mixture:
    # dense-low-noise, sparse-low-noise, dense-high-noise.
    target_region_probs: Tuple[float, float, float] = (0.40, 0.35, 0.25)

    # Same weak-support training mixture as Experiment 1.
    train_region_probs: Tuple[float, float, float] = (0.57, 0.05, 0.38)

    # Data-generating process.
    cluster_std: float = 0.75
    low_noise_std: float = 0.15
    high_noise_std: float = 0.65
    sparse_bump_amplitude: float = 1.00
    sparse_bump_bandwidth: float = 0.80

    # Nominal coverage.
    alpha: float = 0.10

    # Shared heteroscedastic neural model.
    hidden_dim: int = 64
    feature_dim: int = 64
    max_epochs: int = 250
    patience: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    min_delta: float = 1e-5
    variance_floor: float = 1e-4

    # Last-layer posterior.
    lambda_grid: Tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    )
    jitter: float = 1e-6

    # Independent support distances.
    support_k: int = 20
    raw_ood_quantile: float = 0.99
    collapsed_feature_quantile: float = 0.90
    visible_feature_quantile: float = 0.99

    # Candidate construction in standardized raw-input space.
    # The visible condition uses random signal-coordinate shifts. The
    # collapsed condition uses gradient-based input optimization.
    candidate_pool_size: int = 50000
    max_candidate_rounds: int = 6
    min_shift_radius: float = 4.0
    max_shift_radius: float = 8.0

    # Harmful feature-collapse construction. The optimized point must change
    # the true conditional mean by at least this many training-response
    # standard deviations while remaining feature-close to an ID anchor.
    collapsed_min_mean_shift_y_scale: float = 0.75
    collapsed_optimization_pool_size: int = 6000
    collapsed_optimization_batch_size: int = 1000
    collapsed_optimization_steps: int = 100
    collapsed_optimization_lr: float = 3e-2
    collapsed_feature_penalty: float = 1.0
    collapsed_target_reward: float = 1.5
    collapsed_raw_shift_penalty: float = 4.0
    collapsed_max_abs_standardized_input: float = 8.0

    # Paired bootstrap across seeds.
    bootstrap_repeats: int = 5000
    bootstrap_seed: int = 20260801

    # Figure and saving.
    figure_seed: int = 0
    max_figure_points_per_condition: int = 1500
    summary_digits: int = 4
    verbose: bool = True
    display_raw_results: bool = False
    save_csv: bool = True
    output_prefix: str = "ood_harmful_collapse"


CFG = OODConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGION_DENSE_LOW = 0
REGION_SPARSE_LOW = 1
REGION_DENSE_HIGH = 2

CONDITION_ORDER = [
    "ID reference",
    "Feature-visible OOD",
    "Feature-collapsed OOD",
]

METHOD_ORDER = [
    "LACP",
    "Unweighted Laplace",
    "CLAPS",
]

print(f"Using device: {DEVICE}")
print("Appendix experiment: OOD stress test under a fixed representation")
print(
    "Conditions: ID reference, feature-visible OOD, "
    "feature-collapsed OOD"
)


# ============================================================
# Reproducibility and generic utilities
# ============================================================


def ood_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ood_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def ood_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def ood_deterministic_split(
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


def ood_finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    n = len(values)
    if n == 0:
        return np.inf

    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return np.inf

    return float(np.partition(values, rank - 1)[rank - 1])


def ood_interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    width = upper - lower
    score = width.copy()
    below = y < lower
    above = y > upper
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return score


def ood_unstandardize_y(
    y_std: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    return y_std * y_scale + y_mean


def ood_safe_sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def ood_format_mean_se(
    values: Sequence[float],
    digits: int,
    multiplier: float = 1.0,
    suffix: str = "",
) -> str:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return "—"

    mean_value = float(np.mean(array)) * multiplier
    se_value = ood_safe_sem(array) * multiplier
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}{suffix}"


# ============================================================
# Experiment-1 synthetic data-generating process
# ============================================================


def ood_region_centers(input_dim: int) -> np.ndarray:
    centers = np.zeros((3, input_dim), dtype=np.float32)
    centers[REGION_DENSE_LOW, 0:2] = np.array([-2.5, 0.0])
    centers[REGION_SPARSE_LOW, 0:2] = np.array([2.5, 0.0])
    centers[REGION_DENSE_HIGH, 0:2] = np.array([0.0, 2.8])
    return centers


def ood_sample_region_mixture(
    n: int,
    probabilities: Tuple[float, float, float],
    cfg: OODConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / probs.sum()

    region = rng.choice(3, size=n, p=probs)
    centers = ood_region_centers(cfg.input_dim)
    x = centers[region] + cfg.cluster_std * rng.normal(
        size=(n, cfg.input_dim)
    )
    return x.astype(np.float32), region.astype(np.int64)


def ood_true_mean(x_raw: np.ndarray, cfg: OODConfig) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float32)

    base = (
        0.80 * np.sin(x[:, 0:1])
        + 0.35 * x[:, 1:2]
        + 0.20 * np.square(x[:, 2:3])
        - 0.18 * x[:, 0:1] * x[:, 1:2]
        + 0.15 * np.tanh(x[:, 3:4])
    )

    sparse_center = ood_region_centers(cfg.input_dim)[REGION_SPARSE_LOW]
    distance2 = np.sum(
        np.square(x[:, 0:2] - sparse_center[None, 0:2]),
        axis=1,
        keepdims=True,
    )
    local_bump = cfg.sparse_bump_amplitude * np.exp(
        -distance2 / (2.0 * cfg.sparse_bump_bandwidth**2)
    )
    return (base + local_bump).astype(np.float32)


def ood_true_noise_std(
    region: np.ndarray,
    cfg: OODConfig,
) -> np.ndarray:
    sigma = np.full((len(region), 1), cfg.low_noise_std, dtype=np.float32)
    sigma[np.asarray(region) == REGION_DENSE_HIGH] = cfg.high_noise_std
    return sigma


def ood_generate_y(
    x_raw: np.ndarray,
    region: np.ndarray,
    cfg: OODConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = ood_true_mean(x_raw, cfg)
    sigma = ood_true_noise_std(region, cfg)
    y = mean + sigma * rng.normal(size=(len(x_raw), 1))
    return y.astype(np.float32), mean, sigma


def ood_make_base_dataset(
    seed: int,
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(1_000_000 + seed)

    x_train_raw, region_train = ood_sample_region_mixture(
        cfg.n_train,
        cfg.train_region_probs,
        cfg,
        rng,
    )
    x_cal_raw, region_cal = ood_sample_region_mixture(
        cfg.n_cal,
        cfg.target_region_probs,
        cfg,
        rng,
    )
    x_id_raw, region_id = ood_sample_region_mixture(
        cfg.n_id_test,
        cfg.target_region_probs,
        cfg,
        rng,
    )

    y_train, mean_train, sigma_train = ood_generate_y(
        x_train_raw,
        region_train,
        cfg,
        rng,
    )
    y_cal, mean_cal, sigma_cal = ood_generate_y(
        x_cal_raw,
        region_cal,
        cfg,
        rng,
    )
    y_id, mean_id, sigma_id = ood_generate_y(
        x_id_raw,
        region_id,
        cfg,
        rng,
    )

    x_mean = np.mean(x_train_raw, axis=0, keepdims=True)
    x_scale = np.std(x_train_raw, axis=0, keepdims=True) + 1e-6

    x_train = ((x_train_raw - x_mean) / x_scale).astype(np.float32)
    x_cal = ((x_cal_raw - x_mean) / x_scale).astype(np.float32)
    x_id = ((x_id_raw - x_mean) / x_scale).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train) + 1e-8)

    return {
        "x_train": x_train,
        "x_cal": x_cal,
        "x_id": x_id,
        "x_train_raw": x_train_raw,
        "x_cal_raw": x_cal_raw,
        "x_id_raw": x_id_raw,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_id": y_id,
        "y_train_std": ((y_train - y_mean) / y_scale).astype(np.float32),
        "y_cal_std": ((y_cal - y_mean) / y_scale).astype(np.float32),
        "region_train": region_train,
        "region_cal": region_cal,
        "region_id": region_id,
        "mean_train": mean_train,
        "mean_cal": mean_cal,
        "mean_id": mean_id,
        "sigma_train": sigma_train,
        "sigma_cal": sigma_cal,
        "sigma_id": sigma_id,
        "x_mean": x_mean.astype(np.float32),
        "x_scale": x_scale.astype(np.float32),
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


# ============================================================
# Shared heteroscedastic neural regressor
# ============================================================


class OODHeteroMLP(nn.Module):
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


def ood_train_hetero_model(
    x: np.ndarray,
    y_std: np.ndarray,
    seed: int,
    cfg: OODConfig,
) -> OODHeteroMLP:
    ood_set_seed(2_000_000 + seed)

    model = OODHeteroMLP(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        feature_dim=cfg.feature_dim,
        variance_floor=cfg.variance_floor,
    ).to(DEVICE)

    x_t = ood_to_tensor(x)
    y_t = ood_to_tensor(y_std)
    fit_idx, val_idx = ood_deterministic_split(
        len(x),
        cfg.validation_fraction,
        seed=2_100_000 + seed,
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


def ood_predict_hetero(
    model: OODHeteroMLP,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        mu, h2, phi = model(ood_to_tensor(x))
    return (
        ood_to_numpy(mu).reshape(-1, 1),
        ood_to_numpy(h2).reshape(-1, 1),
        ood_to_numpy(phi),
    )


# ============================================================
# Last-layer posterior geometry and prior selection
# ============================================================


def ood_add_bias_column(phi: np.ndarray) -> np.ndarray:
    ones = np.ones((len(phi), 1), dtype=np.float64)
    return np.concatenate([phi.astype(np.float64), ones], axis=1)


def ood_invert_psd(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def ood_compute_weighted_covariance(
    phi_train: np.ndarray,
    h2_train: np.ndarray,
    prior_precision: float,
    cfg: OODConfig,
) -> np.ndarray:
    h_aug = ood_add_bias_column(phi_train)
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
    return ood_invert_psd(precision)


def ood_compute_unweighted_covariance(
    phi_train: np.ndarray,
    prior_precision: float,
    cfg: OODConfig,
) -> np.ndarray:
    h_aug = ood_add_bias_column(phi_train)
    precision = float(prior_precision) * np.eye(
        h_aug.shape[1],
        dtype=np.float64,
    )
    precision += h_aug.T @ h_aug
    precision += cfg.jitter * np.eye(h_aug.shape[1], dtype=np.float64)
    return ood_invert_psd(precision)


def ood_quadratic_form(
    phi: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    h_aug = ood_add_bias_column(phi)
    value = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(value, 0.0).astype(np.float32)


def ood_choose_prior_precision(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    seed: int,
    cfg: OODConfig,
    weighted: bool,
) -> float:
    fit_idx, val_idx = ood_deterministic_split(
        len(data["x_train"]),
        cfg.validation_fraction,
        seed=(2_200_000 if weighted else 2_300_000) + seed,
    )

    _, h2_fit, phi_fit = ood_predict_hetero(
        model,
        data["x_train"][fit_idx],
    )
    mu_val, h2_val, phi_val = ood_predict_hetero(
        model,
        data["x_train"][val_idx],
    )
    y_val = data["y_train_std"][val_idx]

    best_lambda = float(cfg.lambda_grid[0])
    best_nll = float("inf")

    for lam in cfg.lambda_grid:
        if weighted:
            covariance = ood_compute_weighted_covariance(
                phi_fit,
                h2_fit,
                float(lam),
                cfg,
            )
        else:
            covariance = ood_compute_unweighted_covariance(
                phi_fit,
                float(lam),
                cfg,
            )

        epi_val = ood_quadratic_form(phi_val, covariance)
        total_var = np.maximum(h2_val + epi_val, cfg.variance_floor)
        residual2 = (y_val - mu_val) ** 2
        nll = float(
            0.5 * np.mean(residual2 / total_var + np.log(total_var))
        )

        if nll < best_nll:
            best_nll = nll
            best_lambda = float(lam)

    return best_lambda


# ============================================================
# Independent raw-input and learned-feature support distances
# ============================================================


@dataclass
class OODSupportGeometry:
    raw_knn: NearestNeighbors
    feature_knn: NearestNeighbors
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    raw_threshold: float
    collapsed_feature_threshold: float
    visible_feature_threshold: float


def ood_standardize_features(
    phi: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return ((phi - mean) / scale).astype(np.float32)


def ood_mean_knn_distance(
    estimator: NearestNeighbors,
    query: np.ndarray,
) -> np.ndarray:
    distances, _ = estimator.kneighbors(query, return_distance=True)
    return np.mean(distances, axis=1)


def ood_fit_support_geometry(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    cfg: OODConfig,
) -> Tuple[OODSupportGeometry, np.ndarray, np.ndarray]:
    _, _, phi_train = ood_predict_hetero(model, data["x_train"])
    _, _, phi_id = ood_predict_hetero(model, data["x_id"])

    feature_mean = np.mean(phi_train, axis=0, keepdims=True)
    feature_scale = np.std(phi_train, axis=0, keepdims=True)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)

    phi_train_std = ood_standardize_features(
        phi_train,
        feature_mean,
        feature_scale,
    )
    phi_id_std = ood_standardize_features(
        phi_id,
        feature_mean,
        feature_scale,
    )

    n_neighbors = min(max(int(cfg.support_k), 1), len(data["x_train"]))
    raw_knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(data["x_train"])
    feature_knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(phi_train_std)

    raw_id_distance = ood_mean_knn_distance(raw_knn, data["x_id"])
    feature_id_distance = ood_mean_knn_distance(feature_knn, phi_id_std)

    geometry = OODSupportGeometry(
        raw_knn=raw_knn,
        feature_knn=feature_knn,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        raw_threshold=float(
            np.quantile(raw_id_distance, cfg.raw_ood_quantile)
        ),
        collapsed_feature_threshold=float(
            np.quantile(
                feature_id_distance,
                cfg.collapsed_feature_quantile,
            )
        ),
        visible_feature_threshold=float(
            np.quantile(
                feature_id_distance,
                cfg.visible_feature_quantile,
            )
        ),
    )
    return geometry, raw_id_distance, feature_id_distance


def ood_support_distances(
    model: OODHeteroMLP,
    x: np.ndarray,
    geometry: OODSupportGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, phi = ood_predict_hetero(model, x)
    phi_std = ood_standardize_features(
        phi,
        geometry.feature_mean,
        geometry.feature_scale,
    )
    raw_distance = ood_mean_knn_distance(geometry.raw_knn, x)
    feature_distance = ood_mean_knn_distance(
        geometry.feature_knn,
        phi_std,
    )
    return raw_distance, feature_distance, phi


# ============================================================
# OOD candidate construction
# ============================================================


def ood_random_shift(
    n: int,
    active_dimensions: Sequence[int],
    rng: np.random.Generator,
    cfg: OODConfig,
) -> np.ndarray:
    active_dimensions = list(active_dimensions)
    shift = np.zeros((n, cfg.input_dim), dtype=np.float32)
    direction = rng.normal(size=(n, len(active_dimensions))).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-8)
    radius = rng.uniform(
        cfg.min_shift_radius,
        cfg.max_shift_radius,
        size=(n, 1),
    ).astype(np.float32)
    shift[:, active_dimensions] = direction * radius
    return shift


def ood_true_mean_torch(
    x_raw: torch.Tensor,
    cfg: OODConfig,
) -> torch.Tensor:
    """Differentiable version of the synthetic conditional mean."""
    base = (
        0.80 * torch.sin(x_raw[:, 0:1])
        + 0.35 * x_raw[:, 1:2]
        + 0.20 * torch.square(x_raw[:, 2:3])
        - 0.18 * x_raw[:, 0:1] * x_raw[:, 1:2]
        + 0.15 * torch.tanh(x_raw[:, 3:4])
    )

    sparse_center = torch.as_tensor(
        ood_region_centers(cfg.input_dim)[REGION_SPARSE_LOW, 0:2],
        dtype=x_raw.dtype,
        device=x_raw.device,
    ).reshape(1, 2)
    distance2 = torch.sum(
        torch.square(x_raw[:, 0:2] - sparse_center),
        dim=1,
        keepdim=True,
    )
    local_bump = cfg.sparse_bump_amplitude * torch.exp(
        -distance2 / (2.0 * cfg.sparse_bump_bandwidth**2)
    )
    return base + local_bump


def ood_region_quotas(
    n: int,
    probabilities: Sequence[float],
) -> np.ndarray:
    """Integer region counts that sum to n and match the target mixture."""
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / probs.sum()
    raw = n * probs
    quotas = np.floor(raw).astype(int)
    remainder = int(n - quotas.sum())
    if remainder > 0:
        order = np.argsort(-(raw - quotas))
        quotas[order[:remainder]] += 1
    return quotas


def ood_sample_anchor_indices_by_region(
    region_id: np.ndarray,
    pool_size: int,
    desired_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an anchor pool while emphasizing regions with unmet quotas."""
    desired = np.asarray(desired_counts, dtype=float)
    desired = np.maximum(desired, 1.0)
    desired = desired / desired.sum()
    pool_counts = ood_region_quotas(pool_size, desired)

    sampled: List[np.ndarray] = []
    for region, count in enumerate(pool_counts):
        available = np.flatnonzero(np.asarray(region_id) == region)
        if len(available) == 0:
            raise RuntimeError(f"No ID anchors are available for region {region}.")
        sampled.append(
            rng.choice(available, size=int(count), replace=True)
        )
    indices = np.concatenate(sampled)
    return rng.permutation(indices)


def ood_select_region_balanced(
    x: np.ndarray,
    region: np.ndarray,
    raw_distance: np.ndarray,
    feature_distance: np.ndarray,
    mean_shift: np.ndarray,
    anchor_true_mean: np.ndarray,
    candidate_true_mean: np.ndarray,
    rng: np.random.Generator,
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    """Select the same region mixture for every OOD condition."""
    quotas = ood_region_quotas(cfg.n_ood_test, cfg.target_region_probs)
    chosen_parts: List[np.ndarray] = []

    for region_id, quota in enumerate(quotas):
        available = np.flatnonzero(region == region_id)
        if len(available) < quota:
            raise RuntimeError(
                f"Region {region_id} has only {len(available)} valid "
                f"candidates, but {quota} are required. Increase the "
                "candidate pool, optimization rounds, or relax the harmful "
                "feature-collapse threshold slightly."
            )
        chosen_parts.append(
            rng.choice(available, size=int(quota), replace=False)
        )

    chosen = np.concatenate(chosen_parts)
    chosen = rng.permutation(chosen)

    return {
        "x": x[chosen].astype(np.float32),
        "region": region[chosen].astype(np.int64),
        "raw_distance": raw_distance[chosen].astype(np.float32),
        "feature_distance": feature_distance[chosen].astype(np.float32),
        "mean_shift": mean_shift[chosen].astype(np.float32),
        "anchor_true_mean": anchor_true_mean[chosen].astype(np.float32),
        "candidate_true_mean": candidate_true_mean[chosen].astype(np.float32),
    }


def ood_optimize_harmful_collapsed_batch(
    model: OODHeteroMLP,
    x_anchor: np.ndarray,
    data: Mapping[str, np.ndarray],
    geometry: OODSupportGeometry,
    rng: np.random.Generator,
    cfg: OODConfig,
) -> np.ndarray:
    """Optimize inputs to change f(x) while preserving the frozen feature.

    The optimization uses feature closeness to an ID anchor as a differentiable
    proxy. The final candidate still has to satisfy the independent kNN
    feature-distance threshold used in the reported experiment.
    """
    model.eval()
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    x_anchor_t = ood_to_tensor(x_anchor)
    x_mean_t = ood_to_tensor(np.asarray(data["x_mean"], dtype=np.float32))
    x_scale_t = ood_to_tensor(np.asarray(data["x_scale"], dtype=np.float32))
    feature_mean_t = ood_to_tensor(
        np.asarray(geometry.feature_mean, dtype=np.float32)
    )
    feature_scale_t = ood_to_tensor(
        np.asarray(geometry.feature_scale, dtype=np.float32)
    )

    with torch.no_grad():
        phi_anchor = model.features(x_anchor_t)
        phi_anchor_std = (phi_anchor - feature_mean_t) / feature_scale_t
        anchor_raw_t = x_anchor_t * x_scale_t + x_mean_t
        f_anchor_t = ood_true_mean_torch(anchor_raw_t, cfg)

    # Begin with a large signal-coordinate displacement. Nuisance coordinates
    # remain free during optimization and can compensate in feature space.
    initial_shift = ood_random_shift(
        len(x_anchor),
        active_dimensions=list(range(4)),
        rng=rng,
        cfg=cfg,
    )
    x_var = torch.nn.Parameter(
        x_anchor_t + ood_to_tensor(initial_shift)
    )
    optimizer = torch.optim.Adam([x_var], lr=cfg.collapsed_optimization_lr)

    for _ in range(cfg.collapsed_optimization_steps):
        optimizer.zero_grad()
        phi = model.features(x_var)
        phi_std = (phi - feature_mean_t) / feature_scale_t
        feature_mse = torch.mean(
            torch.square(phi_std - phi_anchor_std),
            dim=1,
        )

        x_raw = x_var * x_scale_t + x_mean_t
        f_current = ood_true_mean_torch(x_raw, cfg)
        target_shift = torch.abs(f_current - f_anchor_t).reshape(-1) / max(
            float(data["y_scale"]),
            1e-8,
        )

        anchor_displacement = torch.linalg.vector_norm(
            x_var - x_anchor_t,
            dim=1,
        )
        raw_shift_penalty = torch.square(
            torch.relu(cfg.min_shift_radius - anchor_displacement)
        )

        loss = torch.mean(
            cfg.collapsed_feature_penalty * feature_mse
            - cfg.collapsed_target_reward * target_shift
            + cfg.collapsed_raw_shift_penalty * raw_shift_penalty
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_var.clamp_(
                -cfg.collapsed_max_abs_standardized_input,
                cfg.collapsed_max_abs_standardized_input,
            )

    optimized = ood_to_numpy(x_var).astype(np.float32)

    for parameter, flag in zip(model.parameters(), parameter_flags):
        parameter.requires_grad_(flag)

    return optimized


def ood_collect_visible_candidates(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    geometry: OODSupportGeometry,
    seed: int,
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(3_000_000 + seed)
    selected_x: List[np.ndarray] = []
    selected_region: List[np.ndarray] = []
    selected_raw_distance: List[np.ndarray] = []
    selected_feature_distance: List[np.ndarray] = []
    selected_mean_shift: List[np.ndarray] = []
    selected_anchor_mean: List[np.ndarray] = []
    selected_candidate_mean: List[np.ndarray] = []

    signal_dimensions = list(range(4))
    quotas = ood_region_quotas(cfg.n_ood_test, cfg.target_region_probs)

    for _ in range(cfg.max_candidate_rounds):
        counts = np.zeros(3, dtype=int)
        if selected_region:
            counts = np.bincount(np.concatenate(selected_region), minlength=3)
        deficits = np.maximum(quotas - counts, 0)
        anchor_idx = ood_sample_anchor_indices_by_region(
            data["region_id"],
            cfg.candidate_pool_size,
            np.maximum(deficits, 1),
            rng,
        )
        x_anchor = data["x_id"][anchor_idx]
        x_candidate = x_anchor + ood_random_shift(
            len(x_anchor),
            signal_dimensions,
            rng,
            cfg,
        )
        region_candidate = data["region_id"][anchor_idx]

        raw_distance, feature_distance, _ = ood_support_distances(
            model,
            x_candidate,
            geometry,
        )
        valid = (
            (raw_distance > geometry.raw_threshold)
            & (feature_distance > geometry.visible_feature_threshold)
        )

        if np.any(valid):
            x_candidate_raw = (
                x_candidate[valid] * data["x_scale"] + data["x_mean"]
            )
            candidate_mean = ood_true_mean(x_candidate_raw, cfg).reshape(-1)
            anchor_mean = data["mean_id"][anchor_idx[valid]].reshape(-1)

            selected_x.append(x_candidate[valid])
            selected_region.append(region_candidate[valid])
            selected_raw_distance.append(raw_distance[valid])
            selected_feature_distance.append(feature_distance[valid])
            selected_mean_shift.append(np.abs(candidate_mean - anchor_mean))
            selected_anchor_mean.append(anchor_mean)
            selected_candidate_mean.append(candidate_mean)

        counts = np.zeros(3, dtype=int)
        if selected_region:
            regions_so_far = np.concatenate(selected_region)
            counts = np.bincount(regions_so_far, minlength=3)
        if np.all(counts >= quotas):
            break

    if not selected_x:
        raise RuntimeError(f"No strict visible OOD candidates found for seed={seed}.")

    x_all = np.concatenate(selected_x)
    region_all = np.concatenate(selected_region)
    raw_all = np.concatenate(selected_raw_distance)
    feature_all = np.concatenate(selected_feature_distance)
    mean_shift_all = np.concatenate(selected_mean_shift)
    anchor_mean_all = np.concatenate(selected_anchor_mean)
    candidate_mean_all = np.concatenate(selected_candidate_mean)

    return ood_select_region_balanced(
        x_all,
        region_all,
        raw_all,
        feature_all,
        mean_shift_all,
        anchor_mean_all,
        candidate_mean_all,
        rng,
        cfg,
    )


def ood_collect_harmful_collapsed_candidates(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    geometry: OODSupportGeometry,
    seed: int,
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(4_000_000 + seed)
    selected_x: List[np.ndarray] = []
    selected_region: List[np.ndarray] = []
    selected_raw_distance: List[np.ndarray] = []
    selected_feature_distance: List[np.ndarray] = []
    selected_mean_shift: List[np.ndarray] = []
    selected_anchor_mean: List[np.ndarray] = []
    selected_candidate_mean: List[np.ndarray] = []

    quotas = ood_region_quotas(cfg.n_ood_test, cfg.target_region_probs)
    minimum_mean_shift = (
        cfg.collapsed_min_mean_shift_y_scale * float(data["y_scale"])
    )

    for _ in range(cfg.max_candidate_rounds):
        pool_size = int(cfg.collapsed_optimization_pool_size)
        counts = np.zeros(3, dtype=int)
        if selected_region:
            counts = np.bincount(np.concatenate(selected_region), minlength=3)
        deficits = np.maximum(quotas - counts, 0)
        anchor_idx_all = ood_sample_anchor_indices_by_region(
            data["region_id"],
            pool_size,
            np.maximum(deficits, 1),
            rng,
        )

        optimized_batches: List[np.ndarray] = []
        for start in range(0, pool_size, cfg.collapsed_optimization_batch_size):
            stop = min(start + cfg.collapsed_optimization_batch_size, pool_size)
            batch_idx = anchor_idx_all[start:stop]
            optimized_batches.append(
                ood_optimize_harmful_collapsed_batch(
                    model,
                    data["x_id"][batch_idx],
                    data,
                    geometry,
                    rng,
                    cfg,
                )
            )
        x_candidate = np.concatenate(optimized_batches, axis=0)
        region_candidate = data["region_id"][anchor_idx_all]

        raw_distance, feature_distance, _ = ood_support_distances(
            model,
            x_candidate,
            geometry,
        )
        x_candidate_raw = x_candidate * data["x_scale"] + data["x_mean"]
        candidate_mean = ood_true_mean(x_candidate_raw, cfg).reshape(-1)
        anchor_mean = data["mean_id"][anchor_idx_all].reshape(-1)
        mean_shift = np.abs(candidate_mean - anchor_mean)

        valid = (
            (raw_distance > geometry.raw_threshold)
            & (feature_distance < geometry.collapsed_feature_threshold)
            & (mean_shift >= minimum_mean_shift)
        )

        if np.any(valid):
            selected_x.append(x_candidate[valid])
            selected_region.append(region_candidate[valid])
            selected_raw_distance.append(raw_distance[valid])
            selected_feature_distance.append(feature_distance[valid])
            selected_mean_shift.append(mean_shift[valid])
            selected_anchor_mean.append(anchor_mean[valid])
            selected_candidate_mean.append(candidate_mean[valid])

        counts = np.zeros(3, dtype=int)
        if selected_region:
            regions_so_far = np.concatenate(selected_region)
            counts = np.bincount(regions_so_far, minlength=3)

        if cfg.verbose:
            print(
                "    harmful-collapse valid counts by region: "
                f"{counts.tolist()} / required {quotas.tolist()}"
            )

        if np.all(counts >= quotas):
            break

    if not selected_x:
        raise RuntimeError(
            f"No harmful feature-collapsed candidates were found for seed={seed}. "
            "Increase collapsed_optimization_pool_size or max_candidate_rounds, "
            "or reduce collapsed_min_mean_shift_y_scale slightly."
        )

    x_all = np.concatenate(selected_x)
    region_all = np.concatenate(selected_region)
    raw_all = np.concatenate(selected_raw_distance)
    feature_all = np.concatenate(selected_feature_distance)
    mean_shift_all = np.concatenate(selected_mean_shift)
    anchor_mean_all = np.concatenate(selected_anchor_mean)
    candidate_mean_all = np.concatenate(selected_candidate_mean)

    return ood_select_region_balanced(
        x_all,
        region_all,
        raw_all,
        feature_all,
        mean_shift_all,
        anchor_mean_all,
        candidate_mean_all,
        rng,
        cfg,
    )


def ood_collect_candidates(
    condition: str,
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    geometry: OODSupportGeometry,
    seed: int,
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    if condition == "visible":
        return ood_collect_visible_candidates(
            model,
            data,
            geometry,
            seed,
            cfg,
        )
    if condition == "collapsed":
        return ood_collect_harmful_collapsed_candidates(
            model,
            data,
            geometry,
            seed,
            cfg,
        )
    raise ValueError(f"Unknown candidate condition: {condition}")

def ood_build_test_conditions(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    seed: int,
    cfg: OODConfig,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], OODSupportGeometry]:
    geometry, raw_id_distance, feature_id_distance = ood_fit_support_geometry(
        model,
        data,
        cfg,
    )

    visible = ood_collect_candidates(
        "visible",
        model,
        data,
        geometry,
        seed,
        cfg,
    )
    collapsed = ood_collect_candidates(
        "collapsed",
        model,
        data,
        geometry,
        seed,
        cfg,
    )

    rng = np.random.default_rng(5_000_000 + seed)
    conditions: Dict[str, Dict[str, np.ndarray]] = {}

    conditions["ID reference"] = {
        "x": data["x_id"],
        "x_raw": data["x_id_raw"],
        "region": data["region_id"],
        "y": data["y_id"],
        "true_mean": data["mean_id"],
        "sigma": data["sigma_id"],
        "raw_distance": raw_id_distance.astype(np.float32),
        "feature_distance": feature_id_distance.astype(np.float32),
        "mean_shift": np.zeros(len(data["x_id"]), dtype=np.float32),
        "anchor_true_mean": data["mean_id"].reshape(-1).astype(np.float32),
        "candidate_true_mean": data["mean_id"].reshape(-1).astype(np.float32),
    }

    for name, candidate in [
        ("Feature-visible OOD", visible),
        ("Feature-collapsed OOD", collapsed),
    ]:
        x_std = candidate["x"]
        x_raw = x_std * data["x_scale"] + data["x_mean"]
        y, true_mean, sigma = ood_generate_y(
            x_raw,
            candidate["region"],
            cfg,
            rng,
        )
        conditions[name] = {
            "x": x_std.astype(np.float32),
            "x_raw": x_raw.astype(np.float32),
            "region": candidate["region"],
            "y": y,
            "true_mean": true_mean,
            "sigma": sigma,
            "raw_distance": candidate["raw_distance"],
            "feature_distance": candidate["feature_distance"],
            "mean_shift": candidate["mean_shift"],
            "anchor_true_mean": candidate["anchor_true_mean"],
            "candidate_true_mean": candidate["candidate_true_mean"],
        }

    return conditions, geometry


# ============================================================
# Shared conformal calibration and condition-specific prediction
# ============================================================


@dataclass
class OODCalibratedMethods:
    qhat_lacp: float
    qhat_unweighted: float
    qhat_claps: float
    covariance_unweighted: np.ndarray
    covariance_claps: np.ndarray
    lambda_unweighted: float
    lambda_claps: float


def ood_calibrate_methods(
    model: OODHeteroMLP,
    data: Mapping[str, np.ndarray],
    seed: int,
    cfg: OODConfig,
) -> OODCalibratedMethods:
    lambda_unweighted = ood_choose_prior_precision(
        model,
        data,
        seed,
        cfg,
        weighted=False,
    )
    lambda_claps = ood_choose_prior_precision(
        model,
        data,
        seed,
        cfg,
        weighted=True,
    )

    _, h2_train, phi_train = ood_predict_hetero(
        model,
        data["x_train"],
    )
    mu_cal, h2_cal, phi_cal = ood_predict_hetero(
        model,
        data["x_cal"],
    )

    covariance_unweighted = ood_compute_unweighted_covariance(
        phi_train,
        lambda_unweighted,
        cfg,
    )
    covariance_claps = ood_compute_weighted_covariance(
        phi_train,
        h2_train,
        lambda_claps,
        cfg,
    )

    epi_cal_unweighted = ood_quadratic_form(
        phi_cal,
        covariance_unweighted,
    )
    epi_cal_claps = ood_quadratic_form(
        phi_cal,
        covariance_claps,
    )

    lacp_scale = np.sqrt(np.maximum(h2_cal, cfg.variance_floor))
    unweighted_scale = np.sqrt(
        np.maximum(h2_cal + epi_cal_unweighted, cfg.variance_floor)
    )
    claps_scale = np.sqrt(
        np.maximum(h2_cal + epi_cal_claps, cfg.variance_floor)
    )

    residual = np.abs(data["y_cal_std"] - mu_cal)
    qhat_lacp = ood_finite_sample_quantile(
        residual / lacp_scale,
        cfg.alpha,
    )
    qhat_unweighted = ood_finite_sample_quantile(
        residual / unweighted_scale,
        cfg.alpha,
    )
    qhat_claps = ood_finite_sample_quantile(
        residual / claps_scale,
        cfg.alpha,
    )

    return OODCalibratedMethods(
        qhat_lacp=float(qhat_lacp),
        qhat_unweighted=float(qhat_unweighted),
        qhat_claps=float(qhat_claps),
        covariance_unweighted=covariance_unweighted,
        covariance_claps=covariance_claps,
        lambda_unweighted=float(lambda_unweighted),
        lambda_claps=float(lambda_claps),
    )


def ood_predict_condition(
    model: OODHeteroMLP,
    condition_data: Mapping[str, np.ndarray],
    calibrated: OODCalibratedMethods,
    data: Mapping[str, np.ndarray],
    cfg: OODConfig,
) -> Dict[str, np.ndarray]:
    mu, h2, phi = ood_predict_hetero(model, condition_data["x"])
    epi_unweighted = ood_quadratic_form(
        phi,
        calibrated.covariance_unweighted,
    )
    epi_claps = ood_quadratic_form(
        phi,
        calibrated.covariance_claps,
    )

    scales = {
        "LACP": np.sqrt(np.maximum(h2, cfg.variance_floor)),
        "Unweighted Laplace": np.sqrt(
            np.maximum(h2 + epi_unweighted, cfg.variance_floor)
        ),
        "CLAPS": np.sqrt(
            np.maximum(h2 + epi_claps, cfg.variance_floor)
        ),
    }
    qhats = {
        "LACP": calibrated.qhat_lacp,
        "Unweighted Laplace": calibrated.qhat_unweighted,
        "CLAPS": calibrated.qhat_claps,
    }

    output: Dict[str, np.ndarray] = {
        "mu_std": mu,
        "mu": ood_unstandardize_y(mu, data["y_mean"], data["y_scale"]),
        "h2": h2,
        "epi_unweighted": epi_unweighted,
        "epi_claps": epi_claps,
        "unweighted_epistemic_fraction": (
            epi_unweighted
            / np.maximum(h2 + epi_unweighted, cfg.variance_floor)
        ),
        "claps_epistemic_fraction": (
            epi_claps / np.maximum(h2 + epi_claps, cfg.variance_floor)
        ),
    }

    for method_name in METHOD_ORDER:
        half_width_std = qhats[method_name] * scales[method_name]
        lower_std = mu - half_width_std
        upper_std = mu + half_width_std
        output[f"{method_name}_lower"] = ood_unstandardize_y(
            lower_std,
            data["y_mean"],
            data["y_scale"],
        )
        output[f"{method_name}_upper"] = ood_unstandardize_y(
            upper_std,
            data["y_mean"],
            data["y_scale"],
        )

    return output


# ============================================================
# Metrics
# ============================================================


def ood_compute_condition_rows(
    condition_name: str,
    seed: int,
    condition_data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    calibrated: OODCalibratedMethods,
    cfg: OODConfig,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    y = np.asarray(condition_data["y"]).reshape(-1)
    true_mean = np.asarray(condition_data["true_mean"]).reshape(-1)
    mu = np.asarray(predictions["mu"]).reshape(-1)

    method_rows: List[Dict[str, float]] = []
    method_widths: Dict[str, float] = {}

    for method_name in METHOD_ORDER:
        lower = np.asarray(
            predictions[f"{method_name}_lower"]
        ).reshape(-1)
        upper = np.asarray(
            predictions[f"{method_name}_upper"]
        ).reshape(-1)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        score = ood_interval_score(y, lower, upper, cfg.alpha)
        method_widths[method_name] = float(np.mean(width))

        method_rows.append(
            {
                "condition": condition_name,
                "method": method_name,
                "seed": int(seed),
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(width)),
                "interval_score": float(np.mean(score)),
            }
        )

    diagnostic_row = {
        "condition": condition_name,
        "seed": int(seed),
        "raw_knn_distance": float(
            np.mean(condition_data["raw_distance"])
        ),
        "feature_knn_distance": float(
            np.mean(condition_data["feature_distance"])
        ),
        "mean_function_shift": float(
            np.mean(np.asarray(condition_data["mean_shift"]).reshape(-1))
        ),
        "predicted_aleatoric_scale": float(
            np.mean(np.sqrt(np.maximum(predictions["h2"], cfg.variance_floor)))
        ),
        "true_noise_scale": float(
            np.mean(np.asarray(condition_data["sigma"]).reshape(-1))
        ),
        "dense_low_fraction": float(
            np.mean(np.asarray(condition_data["region"]) == REGION_DENSE_LOW)
        ),
        "sparse_low_fraction": float(
            np.mean(np.asarray(condition_data["region"]) == REGION_SPARSE_LOW)
        ),
        "dense_high_fraction": float(
            np.mean(np.asarray(condition_data["region"]) == REGION_DENSE_HIGH)
        ),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "conditional_mean_rmse": float(
            np.sqrt(np.mean((true_mean - mu) ** 2))
        ),
        "unweighted_epistemic_fraction": float(
            np.mean(predictions["unweighted_epistemic_fraction"])
        ),
        "claps_epistemic_fraction": float(
            np.mean(predictions["claps_epistemic_fraction"])
        ),
        "unweighted_lacp_width_ratio": (
            method_widths["Unweighted Laplace"]
            / max(method_widths["LACP"], 1e-12)
        ),
        "claps_lacp_width_ratio": (
            method_widths["CLAPS"]
            / max(method_widths["LACP"], 1e-12)
        ),
        "lambda_unweighted": calibrated.lambda_unweighted,
        "lambda_claps": calibrated.lambda_claps,
    }
    return diagnostic_row, method_rows


def ood_build_figure_points(
    condition_name: str,
    seed: int,
    condition_data: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    cfg: OODConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(6_000_000 + seed)
    n = len(condition_data["x"])
    keep = np.arange(n)
    if n > cfg.max_figure_points_per_condition:
        keep = rng.choice(
            n,
            size=cfg.max_figure_points_per_condition,
            replace=False,
        )

    return pd.DataFrame(
        {
            "condition": condition_name,
            "seed": int(seed),
            "raw_knn_distance": np.asarray(
                condition_data["raw_distance"]
            )[keep],
            "feature_knn_distance": np.asarray(
                condition_data["feature_distance"]
            )[keep],
            "claps_epistemic_fraction": np.asarray(
                predictions["claps_epistemic_fraction"]
            ).reshape(-1)[keep],
        }
    )


# ============================================================
# One seed and full experiment
# ============================================================


def ood_run_single_seed(
    seed: int,
    cfg: OODConfig,
) -> Tuple[
    List[Dict[str, float]],
    List[Dict[str, float]],
    pd.DataFrame,
    Dict[str, float],
]:
    data = ood_make_base_dataset(seed, cfg)
    model = ood_train_hetero_model(
        data["x_train"],
        data["y_train_std"],
        seed,
        cfg,
    )

    conditions, geometry = ood_build_test_conditions(
        model,
        data,
        seed,
        cfg,
    )
    calibrated = ood_calibrate_methods(
        model,
        data,
        seed,
        cfg,
    )

    diagnostic_rows: List[Dict[str, float]] = []
    method_rows: List[Dict[str, float]] = []
    figure_frames: List[pd.DataFrame] = []

    for condition_name in CONDITION_ORDER:
        predictions = ood_predict_condition(
            model,
            conditions[condition_name],
            calibrated,
            data,
            cfg,
        )
        diagnostic_row, condition_method_rows = ood_compute_condition_rows(
            condition_name,
            seed,
            conditions[condition_name],
            predictions,
            calibrated,
            cfg,
        )
        diagnostic_rows.append(diagnostic_row)
        method_rows.extend(condition_method_rows)

        if seed == cfg.figure_seed:
            figure_frames.append(
                ood_build_figure_points(
                    condition_name,
                    seed,
                    conditions[condition_name],
                    predictions,
                    cfg,
                )
            )

    figure_df = (
        pd.concat(figure_frames, ignore_index=True)
        if figure_frames
        else pd.DataFrame()
    )
    thresholds = {
        "raw_threshold": geometry.raw_threshold,
        "collapsed_feature_threshold": (
            geometry.collapsed_feature_threshold
        ),
        "visible_feature_threshold": geometry.visible_feature_threshold,
    }

    del model, data, conditions, calibrated
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return diagnostic_rows, method_rows, figure_df, thresholds


def ood_run_experiment(
    cfg: OODConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    all_diagnostics: List[Dict[str, float]] = []
    all_methods: List[Dict[str, float]] = []
    figure_df = pd.DataFrame()
    figure_thresholds: Dict[str, float] = {}
    start_time = time.time()

    for seed in range(cfg.num_seeds):
        if cfg.verbose:
            elapsed = time.time() - start_time
            print(
                f"[{seed + 1}/{cfg.num_seeds}] seed={seed} | "
                f"elapsed={elapsed:.1f}s"
            )

        diagnostics, methods, seed_figure_df, thresholds = (
            ood_run_single_seed(seed, cfg)
        )
        all_diagnostics.extend(diagnostics)
        all_methods.extend(methods)

        if seed == cfg.figure_seed:
            figure_df = seed_figure_df
            figure_thresholds = thresholds

    return (
        pd.DataFrame(all_diagnostics),
        pd.DataFrame(all_methods),
        figure_df,
        figure_thresholds,
    )


# ============================================================
# Paper-style summaries
# ============================================================


def ood_build_diagnostic_summary(
    diagnostics_df: pd.DataFrame,
    cfg: OODConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    for condition in CONDITION_ORDER:
        subset = diagnostics_df[diagnostics_df["condition"] == condition]
        rows.append(
            {
                "Condition": condition,
                "Raw kNN distance": ood_format_mean_se(
                    subset["raw_knn_distance"],
                    cfg.summary_digits,
                ),
                "Feature kNN distance": ood_format_mean_se(
                    subset["feature_knn_distance"],
                    cfg.summary_digits,
                ),
                "Response RMSE": ood_format_mean_se(
                    subset["rmse"],
                    cfg.summary_digits,
                ),
                "Conditional-mean RMSE": ood_format_mean_se(
                    subset["conditional_mean_rmse"],
                    cfg.summary_digits,
                ),
                "Mean-function shift": ood_format_mean_se(
                    subset["mean_function_shift"],
                    cfg.summary_digits,
                ),
                "Pred. aleatoric scale (std.)": ood_format_mean_se(
                    subset["predicted_aleatoric_scale"],
                    cfg.summary_digits,
                ),
                "Unweighted epi. frac.": ood_format_mean_se(
                    subset["unweighted_epistemic_fraction"],
                    2,
                    multiplier=100.0,
                    suffix="%",
                ),
                "CLAPS epi. frac.": ood_format_mean_se(
                    subset["claps_epistemic_fraction"],
                    2,
                    multiplier=100.0,
                    suffix="%",
                ),
                "Unweighted/LACP width": ood_format_mean_se(
                    subset["unweighted_lacp_width_ratio"],
                    cfg.summary_digits,
                ),
                "CLAPS/LACP width": ood_format_mean_se(
                    subset["claps_lacp_width_ratio"],
                    cfg.summary_digits,
                ),
            }
        )

    return pd.DataFrame(rows)


def ood_build_method_summary(
    method_results_df: pd.DataFrame,
    cfg: OODConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    for condition in CONDITION_ORDER:
        condition_df = method_results_df[
            method_results_df["condition"] == condition
        ]
        for method in METHOD_ORDER:
            subset = condition_df[condition_df["method"] == method]
            rows.append(
                {
                    "Condition": condition,
                    "Method": method,
                    "Coverage": ood_format_mean_se(
                        subset["coverage"],
                        cfg.summary_digits,
                    ),
                    "Width": ood_format_mean_se(
                        subset["mean_width"],
                        cfg.summary_digits,
                    ),
                    "Interval score": ood_format_mean_se(
                        subset["interval_score"],
                        cfg.summary_digits,
                    ),
                }
            )

    return pd.DataFrame(rows)


def ood_paired_bootstrap_claps_vs_lacp(
    method_results_df: pd.DataFrame,
    cfg: OODConfig,
) -> pd.DataFrame:
    claps = method_results_df[
        method_results_df["method"] == "CLAPS"
    ].copy()
    lacp = method_results_df[
        method_results_df["method"] == "LACP"
    ].copy()
    paired = claps.merge(
        lacp,
        on=["condition", "seed"],
        suffixes=("_claps", "_lacp"),
    )

    paired["coverage_diff"] = (
        paired["coverage_claps"] - paired["coverage_lacp"]
    )
    paired["width_change"] = (
        paired["mean_width_claps"] / paired["mean_width_lacp"] - 1.0
    )
    paired["score_change"] = (
        paired["interval_score_claps"]
        / paired["interval_score_lacp"]
        - 1.0
    )

    metric_specs = [
        ("coverage_diff", "Coverage delta", 1.0, ""),
        ("width_change", "Width delta vs. LACP", 100.0, "%"),
        ("score_change", "Score delta vs. LACP", 100.0, "%"),
    ]
    rows: List[Dict[str, str]] = []

    for condition_index, condition in enumerate(CONDITION_ORDER):
        condition_df = paired[paired["condition"] == condition]

        for metric_index, (metric, label, multiplier, suffix) in enumerate(
            metric_specs
        ):
            differences = condition_df[metric].to_numpy(dtype=float)
            observed = float(np.mean(differences)) * multiplier

            if len(differences) <= 1:
                ci_low = ci_high = observed
            else:
                rng = np.random.default_rng(
                    cfg.bootstrap_seed
                    + 1000 * condition_index
                    + metric_index
                )
                sampled_indices = rng.integers(
                    0,
                    len(differences),
                    size=(cfg.bootstrap_repeats, len(differences)),
                )
                bootstrap_means = (
                    differences[sampled_indices].mean(axis=1) * multiplier
                )
                ci_low, ci_high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )

            rows.append(
                {
                    "Condition": condition,
                    "Effect": label,
                    "Mean": f"{observed:.4f}{suffix}",
                    "95% CI": (
                        f"[{ci_low:.4f}, {ci_high:.4f}]{suffix}"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Scatter plot for the representative seed
# ============================================================


def ood_plot_fixed_representation_stress(
    figure_points_df: pd.DataFrame,
    thresholds: Mapping[str, float],
    cfg: OODConfig,
) -> None:
    if figure_points_df.empty:
        print("No figure points were collected; the scatter plot was skipped.")
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for condition in CONDITION_ORDER:
        subset = figure_points_df[
            figure_points_df["condition"] == condition
        ]
        point_size = (
            12.0
            + 90.0
            * subset["claps_epistemic_fraction"].to_numpy(dtype=float)
        )
        ax.scatter(
            subset["raw_knn_distance"],
            subset["feature_knn_distance"],
            s=point_size,
            alpha=0.35,
            label=condition,
        )

    ax.axvline(
        thresholds["raw_threshold"],
        linestyle="--",
        linewidth=1.0,
        label="ID raw-distance Q99",
    )
    ax.axhline(
        thresholds["collapsed_feature_threshold"],
        linestyle=":",
        linewidth=1.0,
        label="ID feature-distance Q90",
    )
    ax.axhline(
        thresholds["visible_feature_threshold"],
        linestyle="-.",
        linewidth=1.0,
        label="ID feature-distance Q99",
    )

    ax.set_xlabel("Raw-input kNN distance")
    ax.set_ylabel("Learned-feature kNN distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    fig.savefig(
        f"{cfg.output_prefix}_fixed_representation.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{cfg.output_prefix}_fixed_representation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# ============================================================
# Run, summarize, save, and display
# ============================================================


if __name__ == "__main__":
    (
        condition_diagnostics_df,
        method_results_df,
        figure_points_df,
        figure_thresholds,
    ) = ood_run_experiment(CFG)

    diagnostic_summary_df = ood_build_diagnostic_summary(
        condition_diagnostics_df,
        CFG,
    )
    method_summary_df = ood_build_method_summary(
        method_results_df,
        CFG,
    )
    paired_effects_df = ood_paired_bootstrap_claps_vs_lacp(
        method_results_df,
        CFG,
    )

    print("\nFinished the OOD stress test.")
    print(
        "\nCondition diagnostics: mean ± standard error over seeds"
    )
    display(diagnostic_summary_df)

    print("\nMethod performance: mean ± standard error over seeds")
    display(method_summary_df)

    print("\nPaired bootstrap: CLAPS versus LACP")
    display(paired_effects_df)

    print(
        "\nCoverage under the two OOD conditions is descriptive only; "
        "the OOD samples are not exchangeable with the calibration sample."
    )

    if CFG.save_csv:
        condition_diagnostics_df.to_csv(
            f"{CFG.output_prefix}_condition_diagnostics.csv",
            index=False,
        )
        method_results_df.to_csv(
            f"{CFG.output_prefix}_method_results.csv",
            index=False,
        )
        diagnostic_summary_df.to_csv(
            f"{CFG.output_prefix}_diagnostic_summary.csv",
            index=False,
        )
        method_summary_df.to_csv(
            f"{CFG.output_prefix}_method_summary.csv",
            index=False,
        )
        paired_effects_df.to_csv(
            f"{CFG.output_prefix}_paired_effects.csv",
            index=False,
        )
        figure_points_df.to_csv(
            f"{CFG.output_prefix}_scatter_seed{CFG.figure_seed}.csv",
            index=False,
        )

    ood_plot_fixed_representation_stress(
        figure_points_df,
        figure_thresholds,
        CFG,
    )

    if CFG.display_raw_results:
        print("\nRaw condition diagnostics")
        display(condition_diagnostics_df.round(CFG.summary_digits))
        print("\nRaw method results")
        display(method_results_df.round(CFG.summary_digits))
"""Heteroscedastic last-layer Laplace geometry used by CLAPS.

The numerical operations follow the real-data experiment in archive/claps.py.
Inputs are predictions/features from an already fitted heteroscedastic model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def add_bias_column(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")
    ones = np.ones((len(features), 1), dtype=np.float32)
    return np.concatenate([features, ones], axis=1)


def weighted_laplace_covariance(
    features: np.ndarray,
    aleatoric_variance: np.ndarray,
    prior_precision: float,
    *,
    variance_floor: float = 1e-4,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Return (lambda I + H.T diag(1 / h²) H + jitter I)^-1."""
    h_aug = add_bias_column(features).astype(np.float64)
    h2 = np.asarray(aleatoric_variance).reshape(-1)
    if h_aug.shape[0] != h2.size:
        raise ValueError("features and aleatoric_variance lengths differ")
    if prior_precision <= 0 or variance_floor <= 0 or jitter < 0:
        raise ValueError("prior_precision and variance_floor must be positive; jitter nonnegative")
    weights = 1.0 / np.maximum(h2, variance_floor)
    precision = prior_precision * np.eye(h_aug.shape[1], dtype=np.float64)
    precision += h_aug.T @ (h_aug * weights[:, None])
    precision += jitter * np.eye(h_aug.shape[1], dtype=np.float64)
    try:
        return np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(precision)


def epistemic_variance(features: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return the nonnegative quadratic form for each learned feature row."""
    h_aug = add_bias_column(features).astype(np.float64)
    value = np.sum((h_aug @ covariance) * h_aug, axis=1, keepdims=True)
    return np.maximum(value, 0.0).astype(np.float32)


def select_prior_precision(
    features_fit: np.ndarray,
    variance_fit: np.ndarray,
    features_validation: np.ndarray,
    mean_validation: np.ndarray,
    variance_validation: np.ndarray,
    targets_validation: np.ndarray,
    candidates: Sequence[float],
    *,
    variance_floor: float = 1e-4,
    jitter: float = 1e-6,
) -> float:
    """Select lambda by validation NLL from training-only inner splits.

    The caller must keep calibration and test targets out of these arrays.
    On ties, the first candidate wins, as in the experiment code.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    residual2 = (
        np.asarray(targets_validation).reshape(-1, 1)
        - np.asarray(mean_validation).reshape(-1, 1)
    ) ** 2
    h2_val = np.asarray(variance_validation).reshape(-1, 1)
    best_lambda, best_nll = float(candidates[0]), float("inf")
    for candidate in candidates:
        covariance = weighted_laplace_covariance(
            features_fit, variance_fit, float(candidate),
            variance_floor=variance_floor, jitter=jitter,
        )
        epi = epistemic_variance(features_validation, covariance)
        total_var = np.maximum(h2_val + epi, variance_floor)
        nll = 0.5 * np.mean(residual2 / total_var + np.log(total_var))
        if nll < best_nll:
            best_lambda, best_nll = float(candidate), float(nll)
    return best_lambda

"""Split conformal calibration with the CLAPS predictive scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import epistemic_variance, weighted_laplace_covariance


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Match the original experiment's conservative finite-sample quantile."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(np.partition(values, k - 1)[k - 1])


@dataclass
class CLAPSCalibrator:
    """Use pre-fitted heteroscedastic predictions to form CLAPS intervals.

    Means, targets, and variances must all use the same target scale.  For a
    standardized target, supply its training-only mean and scale at prediction.
    """

    covariance: np.ndarray
    variance_floor: float = 1e-4
    alpha: float = 0.1
    quantile_: float | None = None

    @classmethod
    def from_training_features(
        cls,
        features_train: np.ndarray,
        variance_train: np.ndarray,
        prior_precision: float,
        *,
        variance_floor: float = 1e-4,
        jitter: float = 1e-6,
        alpha: float = 0.1,
    ) -> "CLAPSCalibrator":
        covariance = weighted_laplace_covariance(
            features_train, variance_train, prior_precision,
            variance_floor=variance_floor, jitter=jitter,
        )
        return cls(covariance, variance_floor, alpha)

    def scale(self, features: np.ndarray, variance: np.ndarray) -> np.ndarray:
        h2 = np.asarray(variance).reshape(-1, 1)
        epi = epistemic_variance(features, self.covariance)
        if h2.shape != epi.shape:
            raise ValueError("features and variance lengths differ")
        return np.sqrt(np.maximum(h2 + epi, self.variance_floor))

    def calibrate(
        self,
        mean_cal: np.ndarray,
        variance_cal: np.ndarray,
        features_cal: np.ndarray,
        targets_cal: np.ndarray,
    ) -> "CLAPSCalibrator":
        mu = np.asarray(mean_cal).reshape(-1, 1)
        y = np.asarray(targets_cal).reshape(-1, 1)
        scale = self.scale(features_cal, variance_cal)
        if mu.shape != scale.shape or y.shape != scale.shape:
            raise ValueError("calibration arrays must share their sample count")
        self.quantile_ = finite_sample_quantile(np.abs(y - mu) / scale, self.alpha)
        return self

    def predict_interval(
        self,
        mean_test: np.ndarray,
        variance_test: np.ndarray,
        features_test: np.ndarray,
        *,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise RuntimeError("Call calibrate before predict_interval")
        if target_scale <= 0:
            raise ValueError("target_scale must be positive")
        mu = np.asarray(mean_test).reshape(-1, 1)
        scale = self.scale(features_test, variance_test)
        if mu.shape != scale.shape:
            raise ValueError("mean_test and features_test lengths differ")
        lower = (mu - self.quantile_ * scale) * target_scale + target_mean
        upper = (mu + self.quantile_ * scale) * target_scale + target_mean
        return lower, upper

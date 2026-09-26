"""CLAPS last-layer Laplace scaling and split conformal calibration."""

from .conformal import CLAPSCalibrator, finite_sample_quantile
from .geometry import epistemic_variance, select_prior_precision, weighted_laplace_covariance

__all__ = [
    "CLAPSCalibrator",
    "epistemic_variance",
    "finite_sample_quantile",
    "select_prior_precision",
    "weighted_laplace_covariance",
]

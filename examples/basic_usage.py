"""Minimal CLAPS calculation with pre-fitted model outputs.

Replace these illustrative arrays with your model's predictive mean, learned
aleatoric variance, and last-layer features. This is not a paper experiment.
"""

import numpy as np

from claps import CLAPSCalibrator


rng = np.random.default_rng(42)
features_train = rng.normal(size=(80, 3)).astype(np.float32)
variance_train = np.full((80, 1), 0.25, dtype=np.float32)

features_cal = rng.normal(size=(30, 3)).astype(np.float32)
mean_cal = features_cal[:, :1]
variance_cal = np.full((30, 1), 0.25, dtype=np.float32)
targets_cal = mean_cal + rng.normal(scale=0.5, size=(30, 1))

calibrator = CLAPSCalibrator.from_training_features(
    features_train, variance_train, prior_precision=1.0, alpha=0.1
)
calibrator.calibrate(mean_cal, variance_cal, features_cal, targets_cal)

features_test = rng.normal(size=(5, 3)).astype(np.float32)
mean_test = features_test[:, :1]
variance_test = np.full((5, 1), 0.25, dtype=np.float32)
lower, upper = calibrator.predict_interval(mean_test, variance_test, features_test)
print(np.column_stack([lower.reshape(-1), upper.reshape(-1)]))

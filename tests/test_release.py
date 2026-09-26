"""Integrity and numerical-equivalence checks against the preserved source."""

from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

from claps import CLAPSCalibrator, select_prior_precision, weighted_laplace_covariance


ROOT = Path(__file__).resolve().parents[1]


class ReleaseChecks(unittest.TestCase):
    def test_archival_extraction_is_lossless(self) -> None:
        from tools.extract_experiments import EXPECTED_SHA256, RANGES, SOURCE, extract

        raw = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_SHA256)
        lines = raw.splitlines(keepends=True)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            extract(destination)
            for filename, (first, last) in RANGES.items():
                expected = b"".join(lines[first - 1 : last])
                self.assertEqual((destination / filename).read_bytes(), expected)
                self.assertEqual((ROOT / "experiments" / filename).read_bytes(), expected)

    def test_core_agrees_with_original_real_data_experiment(self) -> None:
        source = (ROOT / "archive" / "claps.py").read_text()
        parsed = ast.parse(source)
        wanted = {
            "finite_sample_quantile", "add_bias_column", "quadratic_form",
            "compute_weighted_laplace_covariance", "run_claps", "unstandardize_y",
        }
        definitions = [
            node for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            and 4065 < node.lineno < 5936
            and node.name in wanted
        ]
        namespace = {"np": np, "Dict": Dict, "Tuple": Tuple,
                     "HeteroMLP": object, "Config": object}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "original", "exec"), namespace)

        rng = np.random.default_rng(41)
        train = rng.normal(size=(40, 4)).astype(np.float32)
        cal = rng.normal(size=(30, 4)).astype(np.float32)
        test = rng.normal(size=(10, 4)).astype(np.float32)
        data = {"x_train": train, "x_cal": cal, "x_test": test,
                "y_cal_std": rng.normal(size=(30, 1)).astype(np.float32),
                "y_mean": 2.0, "y_scale": 3.0}
        predictions = {}
        for features in (train, cal, test):
            predictions[id(features)] = (
                rng.normal(size=(len(features), 1)).astype(np.float32),
                rng.uniform(0.1, 1.0, size=(len(features), 1)).astype(np.float32),
                features,
            )
        namespace["predict_hetero"] = lambda model, x: predictions[id(x)]
        namespace["choose_prior_precision"] = lambda model, x, y, cfg: 0.1
        data["y_train_std"] = rng.normal(size=(len(train), 1))
        cfg = SimpleNamespace(variance_floor=1e-4, jitter=1e-6, alpha=0.1)

        original_cov = namespace["compute_weighted_laplace_covariance"](
            train, predictions[id(train)][1], 0.1, cfg
        )
        new_cov = weighted_laplace_covariance(
            train, predictions[id(train)][1], 0.1,
            variance_floor=cfg.variance_floor, jitter=cfg.jitter,
        )
        np.testing.assert_allclose(new_cov, original_cov, rtol=0, atol=0)

        expected = namespace["run_claps"](object(), data, cfg)
        model = CLAPSCalibrator.from_training_features(
            train, predictions[id(train)][1], 0.1, alpha=cfg.alpha,
        )
        mu_cal, h2_cal, phi_cal = predictions[id(cal)]
        model.calibrate(mu_cal, h2_cal, phi_cal, data["y_cal_std"])
        mu_test, h2_test, phi_test = predictions[id(test)]
        actual = model.predict_interval(
            mu_test, h2_test, phi_test, target_mean=2.0, target_scale=3.0
        )
        for found, reference in zip(actual, expected):
            np.testing.assert_allclose(found, reference, rtol=1e-7, atol=1e-7)

    def test_prior_selection_uses_validation_nll(self) -> None:
        features_fit = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        features_val = np.array([[0.0], [0.5]], dtype=np.float32)
        chosen = select_prior_precision(
            features_fit, np.ones((3, 1), dtype=np.float32),
            features_val, np.zeros((2, 1)), np.ones((2, 1)),
            np.array([[0.5], [0.5]]), (0.01, 1.0, 100.0),
        )
        self.assertIn(chosen, (0.01, 1.0, 100.0))


if __name__ == "__main__":
    unittest.main()

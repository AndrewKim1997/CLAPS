# CLAPS

Official code for **CLAPS: Aleatoric-Epistemic Scaling via Last-Layer Laplace for Conformal Regression**, accepted by Transactions on Machine Learning Research (TMLR).

CLAPS combines a learned aleatoric variance with a heteroscedastic last-layer Laplace variance to form the local scale for split conformal regression. The method implementation is in `src/claps/`; the exact Colab-exported code used for the reported experiments is preserved in `archive/claps.py`. The eight standalone scripts in `experiments/` are lossless extracts of its experiment cells. The archival file itself is not a standalone Python program because multiple notebook cells were concatenated into one file.

The manuscript's public citation and OpenReview link will be inserted once the camera-ready version is published. Until then, the experiment names and table numbers below refer to the accepted review manuscript.

## Install and try the method

Use Python 3.10 or newer. In a fresh virtual environment, run:

```bash
python -m pip install -e .
python examples/basic_usage.py
```

The example passes pre-fitted model outputs to `CLAPSCalibrator`: the predictive mean, estimated aleatoric variance, and last-layer features. It then calibrates on held-out observations and returns prediction intervals. The example uses illustrative data and does not reproduce a manuscript result. `select_prior_precision` accepts an inner split of training data only; never use calibration or test targets for this selection.

## Reproduce the experiments

Install the additional dependencies in an environment with a compatible PyTorch installation:

```bash
python -m pip install -e '.[experiments]'
```

Each command below runs the full seed count from the manuscript. The synthetic studies use 30 seeds; the real-data studies use 20. They train neural models and may take substantial time. The real-data scripts fetch eight public benchmark datasets from UCI and scikit-learn when needed, so they require network access. Run commands from the repository root.

```bash
python experiments/01_weak_support.py
python experiments/02_weighted_geometry.py
python experiments/03_epistemic_regimes.py
python experiments/04_real_data.py
python experiments/05_prior_precision.py
python experiments/06_representation_dimension.py
python experiments/07_variance_floor.py
python experiments/08_ood_stress.py
```

Run these scripts independently. They preserve each Colab cell's original configuration, baselines, training, calibration, evaluation, and reporting code. Do not run `archive/claps.py` as one Python script. Experiment 1 corresponds to Tables 1–2; Experiment 2 to Table 3 and Appendix Table 6; Experiment 3 to Figure 1 and Appendix Tables 7–8; Experiment 4 to Tables 4–5 and Appendix Table 9. The prior-precision, representation-dimension, variance-floor, and OOD scripts correspond respectively to Appendix Tables 10–12 and Figure 2/Table 13. Some scripts display results while the later appendix scripts also save CSVs or figures in their configured output locations. See [the reproducibility notes](docs/reproducibility.md) for provenance and limits.

## Verify the source and package

The `archive/claps.py` SHA-256 is `7522c43a5ecef32c59b75b8d7a1b3f21361648f703422a7e1691f389bfbbcc60`. Regenerate the independent scripts from the preserved source with `python tools/extract_experiments.py`. This checks the hash and the syntax of every extracted experiment. After installation, run `python -m unittest discover -s tests -v` to check source integrity and compare the package's CLAPS intervals numerically with the original real-data implementation on fixed synthetic arrays.

The experimental scripts have been separated and syntax checked; the full paper-scale neural training runs and their table values still require an environment with PyTorch, internet access to the original datasets, and a full reproduction run. Python package version ranges are installation bounds, not a record of the original Colab environment.

## License

The repository is released under the MIT License in `LICENSE`. Dataset licenses and terms remain with their respective providers; this repository fetches the datasets rather than redistributing them.

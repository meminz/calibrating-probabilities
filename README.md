# Calibration of Probabilities

**Project report:** [report/Calibrating_Probabilities.pdf](report/Calibrating_Probabilities.pdf)

From-scratch implementations of two post-hoc probability calibration methods - **Platt scaling** (sigmoid, gradient descent) and **isotonic regression** (PAV Algorithm) - evaluated on binary classifiers with a leak-free out-of-fold protocol.

## Repo structure

```
src/
├── __init__.py
├── constants.py                -- artifact paths, model registry, seeds
├── data.py                     -- dataset loaders (Spambase, Adult)
├── pipeline.py                 -- training & calibrator fitting pipeline
├── calibration/
│   ├── __init__.py
│   ├── platt_scaling.py        -- Platt scaler (sigmoid, gradient descent)
│   ├── isotonic_regression.py  -- isotonic calibrator (PAVA algorithm)
│   └── oof.py                  -- out-of-fold prediction (5-fold, stratified)
├── models/
│   ├── __init__.py
│   ├── base.py                 -- logistic regression, random forest factories
│   └── mlp.py                  -- NumPy-only MLP (forward/backprop)
├── metrics/
│   ├── __init__.py
│   ├── scores.py               -- accuracy, log-loss, Brier score
│   └── results.py              -- collect metrics from artifacts -> CSV
└── report/
    ├── __init__.py
    ├── diagrams.py             -- reliability diagrams + ECE (95% CI bins)
    ├── printing.py             -- console tables, calibrator params, ECE bins
    └── latex.py                -- LaTeX table export
main.py                         -- entry point
```

## Installation

Requires Python >= 3.14. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/` and installs the pinned dependencies from `uv.lock`. `uv run` re-syncs automatically if the lockfile changed, so this is mainly first-time setup.

## Usage

All commands go through `uv run`, which executes inside the project venv:

```bash
# full run: train base models, fit calibrators, print tables, export LaTeX, plot figures
uv run main.py

# recompute everything from scratch (ignore existing artifacts)
uv run main.py --force-recompute

# fast iteration: tables only, no figures, no LaTeX export
uv run main.py --no-plot --no-latex

# any output step can be toggled individually
uv run main.py --no-latex
uv run main.py --no-plot
```

Existing artifacts are reused automatically on rerun - only missing ones are recomputed. Pass `--force-recompute` to retrain and refit everything.

Alternative without uv: `source .venv/bin/activate`, then run `python main.py ...` as below.

`--no-plot` and `--no-latex` disable figure generation and LaTeX export; both are on by default.

Trained probabilities and calibrator parameters are cached as `.npz` artifacts in `artifacts/`, metrics are written to `artifacts/results.csv`, figures to `report/figures/`, and LaTeX tables to `report/tables/`.

## What it does

1. **Datasets** - Spambase and Adult (UCI via OpenML), stratified 70/30 split, `seed=42`.
2. **Base models** - logistic regression, random forest, and a small NumPy MLP.
3. **Calibration protocol** - calibrators are fitted on **out-of-fold** predictions (5-fold stratified) of the training set, never on the test set, so the test scores stay untouched for evaluation.
4. **Calibrators** - `PlattScaler` fits `sigmoid(A·s + B)` by gradient descent; `IsotonicCalibrator` fits a monotone step function with the Pool Adjacent Violators algorithm. Both implemented from scratch (no sklearn calibrators).
5. **Metrics** - accuracy, log-loss, and Brier score (raw vs. Platt vs. isotonic vs. isotonic+clip), plus ECE with bin-count sensitivity (10 and 20 bins).
6. **Output** - console tables, reliability diagrams with 95% confidence intervals, and LaTeX tables for the report.

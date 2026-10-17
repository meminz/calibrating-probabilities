from pathlib import Path
import csv

import numpy as np

from src.constants import ARTIFACTS_DIR, MODELS
from .scores import accuracy, brier_score, expected_calibration_error, log_loss

def compute_metrics(datasets, csv_path=ARTIFACTS_DIR / "results.csv") -> list[dict]:
    rows = []

    for ds in datasets:
        split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
        y_train, y_test = split["y_train"], split["y_test"]

        for model_name in MODELS:
            files = {
                "raw":      np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz"),
                "platt":    np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz"),
                "isotonic": np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz"),
            }

            variants = [
                ("raw",      files["raw"],      "oof_proba",             "test_proba"),
                ("platt",    files["platt"],    "oof_calibrated",        "test_calibrated"),
                ("isotonic", files["isotonic"], "oof_calibrated",        "test_calibrated"),
                ("iso+clip", files["isotonic"], "oof_calibrated_clipped", "test_calibrated_clipped"),
            ]

            for vname, f, oof_key, test_key in variants:
                oof_v, test_v = f[oof_key], f[test_key]

                rows.append({
                    "dataset": ds.name, "model": model_name, "variant": vname,
                    "acc_oof": accuracy(y_train, oof_v),
                    "acc_test": accuracy(y_test, test_v),
                    "ll_oof": log_loss(y_train, oof_v),
                    "ll_test": log_loss(y_test, test_v),
                    "brier_oof": brier_score(y_train, oof_v),
                    "brier_test": brier_score(y_test, test_v),
                    "ece": expected_calibration_error(test_v, y_test),
                })

    # csv export
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows

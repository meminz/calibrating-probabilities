from pathlib import Path
import numpy as np
import csv

from src.data import Dataset
from src.diagrams import expected_calibration_error
from src.models import make_logistic_regression, make_random_forest
from src.oof import out_of_fold_predict
from src.platt_scaling import PlattScaler
from src.isotonic_regression import IsotonicCalibrator
from src.metrics import log_loss, brier_score, accuracy

ARTIFACTS_DIR = Path("./artifacts/")
MODELS = {
    "logreg": make_logistic_regression(),
    "rf": make_random_forest()
}

CLIP_ISO = 1e-3


def train_base_models(ds: Dataset) -> None:
    np.savez(
        ARTIFACTS_DIR / f"{ds.name}_split.npz",
        X_train=ds.X_train, X_test=ds.X_test,
        y_train=ds.y_train, y_test=ds.y_test,
    )

    for model_name, model in MODELS.items():
        oof_proba = out_of_fold_predict(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)

        test_proba = model.predict_proba(ds.X_test)[:, 1]

        np.savez(
            ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz",
            oof_proba=oof_proba, test_proba=test_proba,
        )

        oof_acc = accuracy(ds.y_train, oof_proba)
        test_acc = accuracy(ds.y_test, test_proba)

        print(f"{ds.name}/{model_name}: OOF acc={oof_acc:.3f}  test acc={test_acc:.3f}")


def fit_platt(ds: Dataset) -> None:
    split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        scaler = PlattScaler(lr=0.1, n_iter=2000).fit(oof_proba, y_train)
        test_cal = scaler.predict(test_proba)

        np.savez(
            ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz",
            oof_calibrated=scaler.predict(oof_proba),
            test_calibrated=test_cal,
        )

def fit_isotonic(ds: Dataset) -> None:
    split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        iso = IsotonicCalibrator().fit(oof_proba, y_train)
        test_cal = iso.predict(test_proba)
        test_cal_clip = iso.predict(test_proba, clip=CLIP_ISO)

        np.savez(
            ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz",
            oof_calibrated=iso.predict(oof_proba),
            test_calibrated=test_cal,
            oof_calibrated_clipped=iso.predict(oof_proba, clip=CLIP_ISO),
            test_calibrated_clipped=iso.predict(test_proba, clip=CLIP_ISO),
        )


def compute_metrics(datasets, csv_path=ARTIFACTS_DIR / "results.csv") -> list[dict]:
    """Compute the full metrics table from artifacts. This is evaluation
    computation, NOT presentation. Writes CSV, returns rows."""
    rows = []
    for ds in datasets:
        split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
        y_train, y_test = split["y_train"], split["y_test"]
        for model_name in ["logreg", "rf"]:
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
                })

    # csv export
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows

def print_tables(rows: list[dict]) -> None:
    # TODO replace with latex exporter
    header = (f"{'variant':9s} {'acc_oof':>7s} {'acc_test':>8s} "
              f"{'ll_oof':>7s} {'ll_test':>7s} {'br_oof':>7s} {'br_test':>7s}")

    for ds_name in dict.fromkeys(r["dataset"] for r in rows):
        print(f"\n{ds_name}\n{header}")

        for r in rows:
            if r["dataset"] == ds_name:
                print(f"{r['variant']:9s} {r['acc_oof']:7.3f} {r['acc_test']:8.3f} "
                      f"{r['ll_oof']:7.4f} {r['ll_test']:7.4f} "
                      f"{r['brier_oof']:7.4f} {r['brier_test']:7.4f}")


def print_ece_sensitivity(datasets, bin_counts=(10, 20)) -> None:
    print("\nECE (test) -- bin-count sensitivity")

    for ds in datasets:
        y_test = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")["y_test"]

        for model_name in ["logreg", "rf"]:
            d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            variants = [("raw", d["test_proba"]),
                        ("iso+clip", iso["test_calibrated_clipped"])]

            for vname, arr in variants:
                eces = "  ".join(f"M={m}: {expected_calibration_error(arr, y_test, m):.4f}"
                                 for m in bin_counts)

                print(f"  {ds.name}/{model_name} [{vname}]  {eces}")
                print(f"  {ds.name}/{model_name} [{vname}]  distinct={len(np.unique(arr))}  {eces}")

            print()

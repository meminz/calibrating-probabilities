from pathlib import Path
import numpy as np
import csv

from src.data import Dataset
from src.models import make_logistic_regression, make_random_forest, make_mlp
from src.oof import out_of_fold_predict
from src.platt_scaling import PlattScaler
from src.isotonic_regression import IsotonicCalibrator
from src.metrics import log_loss, brier_score, accuracy

ARTIFACTS_DIR = Path("./artifacts/")
MODELS = {
    "logreg": make_logistic_regression(),
    "rf": make_random_forest(),
    "mlp": make_mlp(),
}

CLIP_ISO = 1e-3


def train_base_models(ds: Dataset, force_recompute: bool = False) -> None:
    np.savez(
        ARTIFACTS_DIR / f"{ds.name}_split.npz",
        X_train=ds.X_train, X_test=ds.X_test,
        y_train=ds.y_train, y_test=ds.y_test,
    )


    for model_name, model in MODELS.items():
        artifact = ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz"
        if artifact.exists() and not force_recompute:
            print(f"{ds.name}_{model_name}: artifacts already exists, skipping fit")
            continue

        oof_proba = out_of_fold_predict(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)

        test_proba = model.predict_proba(ds.X_test)[:, 1]

        np.savez(
            artifact,
            oof_proba=oof_proba, test_proba=test_proba,
        )

        oof_acc = accuracy(ds.y_train, oof_proba)
        test_acc = accuracy(ds.y_test, test_proba)

        print(f"{ds.name}/{model_name}: OOF acc={oof_acc:.3f}  test acc={test_acc:.3f}")


def fit_platt(ds: Dataset, force_recompute: bool = False) -> None:
    split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        artifact = ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz"
        if artifact.exists() and not force_recompute:
            print(f"{ds.name}_{model_name}: artifacts already exists, skipping fit")
            continue

        d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        scaler = PlattScaler(lr=0.1, n_iter=2000).fit(oof_proba, y_train)
        test_cal = scaler.predict(test_proba)

        np.savez(
            artifact,
            oof_calibrated=scaler.predict(oof_proba),
            test_calibrated=test_cal,
            A=scaler.A_, B=scaler.B_,
        )

def fit_isotonic(ds: Dataset, force_recompute: bool = False) -> None:
    split = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        artifact = ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz"
        if artifact.exists() and not force_recompute:
            print(f"{ds.name}_{model_name}: artifacts already exists, skipping fit")
            continue

        d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        iso = IsotonicCalibrator().fit(oof_proba, y_train)
        test_cal = iso.predict(test_proba)
        test_cal_clip = iso.predict(test_proba, clip=CLIP_ISO)

        np.savez(
            artifact,
            oof_calibrated=iso.predict(oof_proba),
            test_calibrated=test_cal,
            oof_calibrated_clipped=iso.predict(oof_proba, clip=CLIP_ISO),
            test_calibrated_clipped=iso.predict(test_proba, clip=CLIP_ISO),
            n_blocks=len(iso.block_values_),
        )


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
                })

    # csv export
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


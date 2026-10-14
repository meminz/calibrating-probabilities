from pathlib import Path

import numpy as np

from .calibration import IsotonicCalibrator, PlattScaler, out_of_fold_predict
from .constants import ARTIFACTS_DIR, MODELS
from .data import Dataset
from .metrics.scores import accuracy, brier_score, log_loss

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




import numpy as np

from src.data import Dataset
from src.models import make_logistic_regression, make_random_forest
from src.prediction import out_of_fold_predict
from src.platt_scaling import PlattScaler
from src.isotonic_regression import IsotonicCalibrator
from src.metrics import log_loss, brier_score, accuracy

ARTIFACTS_DIR = "./artifacts/"
MODELS = {
    "logreg": make_logistic_regression(),
    "rf": make_random_forest()
}


def train_base_models(ds: Dataset) -> None:
    np.savez(
        ARTIFACTS_DIR + f"{ds.name}_split.npz",
        X_train=ds.X_train, X_test=ds.X_test,
        y_train=ds.y_train, y_test=ds.y_test,
    )

    print()
    for model_name, model in MODELS.items():
        oof_proba = out_of_fold_predict(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)

        test_proba = model.predict_proba(ds.X_test)[:, 1]

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz",
            oof_proba=oof_proba, test_proba=test_proba,
        )

        oof_acc = accuracy(ds.y_train, oof_proba)
        test_acc = accuracy(ds.y_test, test_proba)

        print(f"{ds.name}/{model_name}: OOF acc={oof_acc:.3f}  test acc={test_acc:.3f}")


def fit_platt(ds: Dataset) -> None:
    split = np.load(ARTIFACTS_DIR + f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        d = np.load(ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        scaler = PlattScaler(lr=0.1, n_iter=2000).fit(oof_proba, y_train)
        test_cal = scaler.predict(test_proba)

        print(f"{ds.name}/{model_name}: A={scaler.A_:.3f} B={scaler.B_:.3f} | "
              f"OOF log loss = {log_loss(y_train, oof_proba):.4f} "
              f"platt={log_loss(y_train, scaler.predict(oof_proba)):.4f} | "
              f"test raw={log_loss(y_test, test_proba):.4f} | "
              f"platt={log_loss(y_test, test_cal):.4f} | "
              f"test brier raw {brier_score(y_test, test_proba)}")

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}_platt.npz",
            oof_calibrated=scaler.predict(oof_proba),
            test_calibrated=test_cal,
        )

def fit_isotonic(ds: Dataset) -> None:
    split = np.load(ARTIFACTS_DIR + f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in MODELS:
        d = np.load(ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        iso = IsotonicCalibrator().fit(oof_proba, y_train)
        test_cal = iso.predict(test_proba)
        test_cal_clip = iso.predict(test_proba, clip=1e-3)

        print(f"{ds.name}/{model_name} [iso]: {len(iso.block_values_)} blocks | "
              f"OOF log loss {log_loss(y_train, oof_proba):.4f}->{log_loss(y_train, iso.predict(oof_proba)):.4f} | "
              f"test log loss {log_loss(y_test, test_proba):.4f}->{log_loss(y_test, test_cal):.4f}"
              f" (clip {log_loss(y_test, test_cal_clip):.4f}) | "
              f"test brier raw {brier_score(y_test, test_proba):.4f} | "
              f"iso {brier_score(y_test, test_cal):.4f} | "
              f"(clip {brier_score(y_test, test_cal_clip):.4f})")

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}_isotonic.npz",
            oof_calibrated=iso.predict(oof_proba),
            test_calibrated=test_cal,
        )

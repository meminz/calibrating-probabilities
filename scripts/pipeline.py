import numpy as np

from src.data import Dataset
from src.models import make_logistic_regression, make_random_forest
from src.prediction import out_of_fold_predict
from src.platt_scaling import PlattScaler

ARTIFACTS_DIR = "./artifacts/"


def train_base_models(ds: Dataset) -> None:
    np.savez(
        ARTIFACTS_DIR + f"{ds.name}_split.npz",
        X_train=ds.X_train, X_test=ds.X_test,
        y_train=ds.y_train, y_test=ds.y_test,
    )

    for model_name, model in [
        ("logreg", make_logistic_regression()),
        ("rf", make_random_forest()),
    ]:
        oof_proba = out_of_fold_predict(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)

        test_proba = model.predict_proba(ds.X_test)[:, 1]

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz",
            oof_proba=oof_proba, test_proba=test_proba,
        )

        oof_acc = ((oof_proba >= 0.5).astype(int) == ds.y_train).mean()
        test_acc = ((test_proba >= 0.5).astype(int) == ds.y_test).mean()

        print(f"{ds.name}/{model_name}: OOF acc={oof_acc:.3f}  test acc={test_acc:.3f}")


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    # TODO replace with actual metrics
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_platt(ds: Dataset) -> None:
    split = np.load(ARTIFACTS_DIR + f"{ds.name}_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in ["logreg", "rf"]:
        d = np.load(ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        scaler = PlattScaler(lr=0.1, n_iter=2000).fit(oof_proba, y_train)
        test_cal = scaler.predict(test_proba)

        print(f"{ds.name}/{model_name}: A={scaler.A_:.3f} B={scaler.B_:.3f} | "
              f"OOF raw={log_loss(y_train, oof_proba):.4f} "
              f"platt={log_loss(y_train, scaler.predict(oof_proba)):.4f} | "
              f"test raw={log_loss(y_test, test_proba):.4f} "
              f"platt={log_loss(y_test, test_cal):.4f}")

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}_platt.npz",
            oof_calibrated=scaler.predict(oof_proba),
            test_calibrated=test_cal,
        )

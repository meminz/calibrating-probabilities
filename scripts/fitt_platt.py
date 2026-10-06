"""
fit Platt scaling on OOF scores, evaluate on the test set

fits only on (oof_probabilities, y_train), test data is used for evaluation only
"""
import numpy as np

from src.platt_scaling import PlattScaler

ARTIFACTS_DIR = "./artifacts/"


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    # TODO replace with actual metrics
    # fine to quickly see effects
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    split = np.load(ARTIFACTS_DIR + "breast_cancer_split.npz")
    y_train, y_test = split["y_train"], split["y_test"]

    for model_name in ["logreg", "rf"]:
        d = np.load(ARTIFACTS_DIR + f"breast_cancer_{model_name}.npz")
        oof_proba, test_proba = d["oof_proba"], d["test_proba"]

        print(f"\n=== {model_name} ===")
        scaler = PlattScaler(lr=0.1, n_iter=2000, verbose=True)
        scaler.fit(oof_proba, y_train)
        print(f"  fitted: A={scaler.A_:.3f}  B={scaler.B_:.3f}")

        test_cal = scaler.predict(test_proba)
        print(f"  test log-loss  raw={log_loss(y_test, test_proba):.4f}"
              f"  platt={log_loss(y_test, test_cal):.4f}")

        np.savez(
            ARTIFACTS_DIR + f"breast_cancer_{model_name}_platt.npz",
            oof_calibrated=scaler.predict(oof_proba),
            test_calibrated=test_cal,
        )


if __name__ == "__main__":
    main()

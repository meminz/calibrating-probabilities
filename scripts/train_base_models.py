"""
train base models and cache OOF + test predictions

artifacts (.npz) are the single source of truth for every later step:
calibration, metrics, and plots all load these files instead of retraining
"""
import numpy as np

from src.data import load_breast_cancer_dataset
from src.models import make_logistic_regression, make_random_forest
from src.prediction import out_of_fold_predict

ARTIFACTS_DIR = "./artifacts/"


def main() -> None:
    ds = load_breast_cancer_dataset()

    # cache frozen splits
    np.savez(
        ARTIFACTS_DIR + f"{ds.name}_split.npz",
        X_train=ds.X_train, X_test=ds.X_test,
        y_train=ds.y_train, y_test=ds.y_test,
    )

    for model_name, model in [
        ("logreg", make_logistic_regression(max_iter=2000)),
        ("rf", make_random_forest(n_estimators=1000)),
        # TODO add mlp
    ]:
        oof_proba = out_of_fold_predict(model, ds.X_train, ds.y_train)

        # refit on the full training set
        # this model (with calibrator fitted on the OOF scores) is what gets evaluated on the test set
        model.fit(ds.X_train, ds.y_train)
        test_proba = model.predict_proba(ds.X_test)[:, 1]

        np.savez(
            ARTIFACTS_DIR + f"{ds.name}_{model_name}.npz",
            oof_proba=oof_proba, test_proba=test_proba,
        )

        oof_acc = ((oof_proba >= 0.5).astype(int) == ds.y_train).mean()
        test_acc = ((test_proba >= 0.5).astype(int) == ds.y_test).mean()
        print(f"\n{model_name}  OOF acc={oof_acc:.5f}   test acc={test_acc:.5f}")


if __name__ == "__main__":
    main()

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def out_of_fold_predict(model, X: np.ndarray, y: np.ndarray,
                        n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """
    return out-of-fold predicted probabilities of the positive class (label 1)

    every sample is predicted by a model that never saw it during training,
    this is what makes the scores valid input for fitting a calibrator afterwards
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=float)
    for fit_idx, val_idx in skf.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X[fit_idx], y[fit_idx])
        oof[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
    return oof

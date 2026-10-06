from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.3


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_breast_cancer_dataset(seed: int = RANDOM_SEED,
                               test_size: float = TEST_SIZE) -> Dataset:
    bunch = load_breast_cancer()
    X, y = bunch.data, bunch.target
    # sklearn encodes 0 = malignant, 1 = benign; flip so the positive
    # class (label 1) is malignant -> proba[:, 1] is a malignancy risk score.
    y = (y == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    return Dataset("breast_cancer", X_train, X_test, y_train, y_test)

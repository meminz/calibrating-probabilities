from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TEST_SIZE = 0.3


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_breast_cancer_dataset(seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> Dataset:
    bunch = load_breast_cancer()
    X, y = bunch.data, bunch.target

    # label flip so 1 is malignant (malignancy risk score)
    y = (y == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=seed, stratify=y )

    return Dataset("breast_cancer", X_train, X_test, y_train, y_test)

def load_adult(n: int = None, seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> Dataset:
    """
    adult income binarized to >50K vs <=50K

    one-hot encoding on training set
    handle_unknown='ignore' lets the encoder survive categories that appear only in the test fold.
    """
    frame = fetch_openml("adult", version=2, as_frame=True).frame
    frame = frame.replace("?", np.nan).dropna()
    y_all = (frame["class"].str.strip(". ") == ">50K").astype(int)
    X_all = frame.drop(columns="class")

    if n is not None:
        name = "adult_sub"
        # used as sampler to ensure calss ratio
        _, X_all, _, y_all = train_test_split( X_all, y_all, test_size=n, random_state=seed, stratify=y_all )

    else:
        name = "adult"

    X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all )




    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return Dataset(name, X_train, X_test, y_train, y_test)

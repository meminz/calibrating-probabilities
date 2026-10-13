from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TEST_SIZE = 0.3

NUMERIC_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def load_spambase(seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> Dataset:
    frame = fetch_openml("spambase", version=1, as_frame=True).frame
    y = (frame["class"].astype(str) == "1").astype(int)  # check .unique() once
    X = frame.drop(columns="class").to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    return Dataset("spambase", X_train, X_test,
                   np.asarray(y_train), np.asarray(y_test))

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

    X_all[NUMERIC_COLS] = X_all[NUMERIC_COLS].apply(pd.to_numeric)

    if n is not None:
        name = "adult_sub"
        # used as sampler to ensure calss ratio
        _, X_all, _, y_all = train_test_split( X_all, y_all, test_size=n, random_state=seed, stratify=y_all )

    else:
        name = "adult"

    X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all )

    cat_cols = X_train.select_dtypes(include="object").columns
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train = np.hstack([enc.fit_transform(X_train[cat_cols]), X_train[NUMERIC_COLS].to_numpy()])
    X_test = np.hstack([enc.transform(X_test[cat_cols]), X_test[NUMERIC_COLS].to_numpy()])
    X_train, X_test = map(np.asarray, (X_train, X_test))

    assert X_train.shape[1] < 200

    return Dataset(name, X_train, X_test, np.asarray(y_train), np.asarray(y_test))

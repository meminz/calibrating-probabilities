import numpy as np

EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    assert isinstance(y, np.ndarray), "y must be np.ndarray"
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)

    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    assert isinstance(y, np.ndarray), "y must be np.ndarray"
    assert isinstance(p, np.ndarray), "p must be np.ndarray"

    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    assert isinstance(y, np.ndarray), "y must be np.ndarray"
    assert isinstance(p, np.ndarray), "p must be np.ndarray"

    return float(np.mean((p >= threshold).astype(int) == y))

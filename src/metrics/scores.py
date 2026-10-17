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


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # bin b owns [edges[b], edges[b+1])
    assignments = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = assignments == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        total += n_b * abs(float(p[mask].mean()) - float(y[mask].mean()))

    return float(total / len(p))

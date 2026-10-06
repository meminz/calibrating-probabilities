import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)

    # ensure numerical stability by branching z<0 and z>=0
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos])) # explodes for z=-inf
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez) # explodes for z=+inf

    return out


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    out[pos] = -np.log1p(np.exp(-z[pos]))
    out[~pos] = z[~pos] - np.log1p(np.exp(z[~pos]))

    return out


class PlattScaler:
    """fit A, B of sigmoid(A*s + B) on (scores, labels) via gradient descent"""

    def __init__(self, lr: float = 0.1, n_iter: int = 2000,
                 l2: float = 1e-3, verbose: bool = False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2 # ridge penalty, Platt's paper used a prior too
        self.verbose = verbose

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattScaler:
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)

        # standardize the input for well-conditioned gradient descent
        mu, sd = scores.mean(), scores.std()
        z = (scores - mu) / sd

        A, B = 1.0, 0.0 # init to identity slope, zero intercept
        for it in range(self.n_iter):
            t = A * z + B
            p = _sigmoid(t)
            err = p - y
            grad_A = np.mean(err * z) + self.l2 * A
            grad_B = np.mean(err)
            A -= self.lr * grad_A
            B -= self.lr * grad_B

            if self.verbose and it % 200 == 0:
                loss = -np.mean(y * _log_sigmoid(t) + (1 - y) * _log_sigmoid(-t))
                print(f"    iter {it:5d}  log-loss {loss:.4f}  A {A:.3f}  B {B:.3f}")

        # map the fitted coefficients back to raw-score space
        self.A_ = A / sd
        self.B_ = B - A * mu / sd
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        return _sigmoid(self.A_ * scores + self.B_)

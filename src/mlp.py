import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0

    out[pos] = 1.0 / (1.0 + np.exp( -z[pos] ))
    exp_z = np.exp(z[~pos])

    out[~pos] = exp_z / (1.0 + exp_z)

    return out


class MLP:

    def __init__(self, n_hidden: int = 16, lr: float = 0.2, n_epochs: int = 300, l2: float = 1e-4, seed: int = 42, verbose: bool = False):
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose


    def get_params(self, deep: bool = True) -> dict:
        return dict(
            n_hidden = self.n_hidden,
            lr = self.lr,
            n_epochs = self.n_epochs,
            l2 = self.l2,
            seed = self.seed,
            verbose = self.verbose
        )
    
    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k, v)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLP:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        y = y.reshape(-1,1)

        n, d = X.shape

        self._mu = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-12

        X_standardized = (X - self._mu) / self._std

        rng = np.random.default_rng(self.seed)

        self.W1_ = rng.uniform(-np.sqrt(1.0 / d), np.sqrt(1.0 / d), size=(d, self.n_hidden))
        self.b1_ = np.zeros( (1, self.n_hidden) )

        self.W2_ = rng.uniform(-np.sqrt(1.0 / self.n_hidden), np.sqrt(1.0 / self.n_hidden), size=(self.n_hidden, 1))
        self.b2_ = np.zeros( (1, 1) )

        for epoch in range(self.n_epochs):
            # forward pass
            a1 = np.tanh(X_standardized @ self.W1_ + self.b1_)
            p = _sigmoid(a1 @ self.W2_ + self.b2_)

            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

            loss += self.l2 * (np.sum(self.W1_ ** 2) + np.sum(self.W2_ ** 2))


            # backward pass
            d_z2 = (p - y) / n
            d_W2 = a1.T @ d_z2 + 2 * self.l2 * self.W2_
            d_b2 = d_z2.sum(axis=0, keepdims=True)

            d_z1 = (d_z2 @ self.W2_.T) * (1 - a1 ** 2)
            d_W1 = X_standardized.T @ d_z1 + 2 * self.l2 * self.W1_
            d_b1 = d_z1.sum(axis=0, keepdims=True)

            self.W1_ -= self.lr * d_W1
            self.b1_ -= self.lr * d_b1
            self.W2_ -= self.lr * d_W2
            self.b2_ -= self.lr * d_b2

            if self.verbose and epoch % 25 == 0:
                print(f"    epoch {epoch:4d}  loss {loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        X_standardized = (X - self._mu) / self._std

        a1 = np.tanh(X_standardized @ self.W1_ + self.b1_)
        p1 = _sigmoid(a1 @ self.W2_ + self.b2_).ravel()

        return np.column_stack([1.0 - p1, p1])


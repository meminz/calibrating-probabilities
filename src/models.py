from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.mlp import MLP

SEED = 42


def make_logistic_regression(max_iter: int = 3000) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=SEED)

def make_random_forest(n_estimators: int = 500, jobs: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=jobs, random_state=SEED)

def make_mlp(n_hidden: int = 16, lr: float = 0.2, n_epochs: int = 300, l2: float = 1e-4, seed: int = 42, verbose: bool = False) -> MLP:
    return MLP(n_hidden=n_hidden, lr=lr, n_epochs=n_epochs, l2=l2, seed=seed, verbose=verbose)

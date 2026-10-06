from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42


def make_logistic_regression(max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=SEED)


def make_random_forest(n_estimators: int = 500 ) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=SEED)

# TODO implement small mlp

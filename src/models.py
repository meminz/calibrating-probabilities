from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=SEED)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)

# TODO implement small mlp

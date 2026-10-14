from pathlib import Path
from src.models import make_logistic_regression, make_random_forest, make_mlp

ARTIFACTS_DIR = Path("./artifacts/")
FIGURES_DIR = Path("./report/figures")

MODELS = {
    "logreg": make_logistic_regression(),
    "rf": make_random_forest(),
    "mlp": make_mlp(),
}

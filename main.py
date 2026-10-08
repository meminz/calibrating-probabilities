from src.data import load_adult_subset, load_breast_cancer_dataset
from src.pipeline import compute_metrics, fit_isotonic, fit_platt, print_tables, train_base_models

DATASETS = [
        load_breast_cancer_dataset(),
        load_adult_subset()
]

def main() -> None:
    for ds in DATASETS:
        train_base_models(ds)
        fit_platt(ds)
        fit_isotonic(ds)

    rows = compute_metrics(DATASETS) # load from artifacts dir
    print_tables(rows)


if __name__ == "__main__":
    main()

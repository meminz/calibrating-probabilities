from argparse import ArgumentParser

from src.data import load_adult_subset, load_breast_cancer_dataset
from src.diagrams import plot_all_diagrams
from src.pipeline import compute_metrics, fit_isotonic, fit_platt, print_ece_sensitivity, print_tables, train_base_models

DATASETS = [
        load_breast_cancer_dataset(),
        load_adult_subset()
]

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", action='store_true', help="print results and compute plots, skip training and fitting")
    args = parser.parse_args()

    if not args.plot:
        for ds in DATASETS:
            train_base_models(ds)
            fit_platt(ds)
            fit_isotonic(ds)

    rows = compute_metrics(DATASETS) # load from artifacts dir
    print_tables(rows)
    plot_all_diagrams(DATASETS)
    print_ece_sensitivity(DATASETS)


if __name__ == "__main__":
    main()

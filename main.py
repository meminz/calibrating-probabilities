from argparse import ArgumentParser

from src.data import load_adult, load_spambase
from src.diagrams import plot_all_diagrams
from src.pipeline import compute_metrics, fit_isotonic, fit_platt, train_base_models
from src.prints import print_calibrator_params, print_ece_sensitivity, print_tables

DATASETS = [
        load_spambase(),
        # load_adult(n=4000),
        load_adult(),
]

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-f", "--force-recompute", action='store_true', help="recompute training and fitting, skip artifacts presence check")
    parser.add_argument("-p", "--plot", action='store_true', help="print results and compute plots, skip training and fitting")
    args = parser.parse_args()

    if not args.plot:
        for ds in DATASETS:
            train_base_models(ds, args.force_recompute)
            fit_platt(ds, args.force_recompute)
            fit_isotonic(ds, args.force_recompute)

    rows = compute_metrics(DATASETS) # load from artifacts dir

    print_tables(rows)
    print_calibrator_params(DATASETS)
    print_ece_sensitivity(DATASETS)

    plot_all_diagrams(DATASETS)


if __name__ == "__main__":
    main()

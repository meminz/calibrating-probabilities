from pathlib import Path
import numpy as np

from .diagrams import expected_calibration_error

ARTIFACTS_DIR = Path("./artifacts/")
MODELS = {
    "logreg",
    "rf",
    "mlp",
}

def print_tables(rows: list[dict]) -> None:
    # TODO replace with latex exporter
    header = (f"{'model':9s} {'variant':9s} {'acc_oof':>7s} {'acc_test':>8s} "
              f"{'ll_oof':>7s} {'ll_test':>7s} {'br_oof':>7s} {'br_test':>7s}")

    for ds_name in dict.fromkeys(r["dataset"] for r in rows):
        print(f"\n{ds_name}\n{header}")

        for r in rows:
            if r["dataset"] == ds_name:
                print(f"{r['model']:9s} {r['variant']:9s} {r['acc_oof']:7.3f} {r['acc_test']:8.3f} "
                      f"{r['ll_oof']:7.4f} {r['ll_test']:7.4f} "
                      f"{r['brier_oof']:7.4f} {r['brier_test']:7.4f}")


def print_ece_sensitivity(datasets, bin_counts=(10, 20)) -> None:
    print("\nECE (test) -- bin-count sensitivity")

    for ds in datasets:
        y_test = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")["y_test"]

        for model_name in MODELS:
            d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            variants = [("raw", d["test_proba"]),
                        ("iso+clip", iso["test_calibrated_clipped"])]

            for vname, arr in variants:
                eces = "  ".join(f"M={m}: {expected_calibration_error(arr, y_test, m):.4f}"
                                 for m in bin_counts)

                print(f"  {ds.name}/{model_name} [{vname}]  distinct={len(np.unique(arr))}  {eces}")

            print()

def print_calibrator_params(datasets) -> None:
    print("\ncalibrator parameters")

    for ds in datasets:
        for model_name in MODELS:
            pl = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            print(f"  {ds.name}/{model_name}: platt A={pl['A']:.3f} B={pl['B']:.3f}"
                  f"  | iso blocks={int(iso['n_blocks'])}")

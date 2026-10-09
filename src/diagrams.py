from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

Z_95 = 1.96  # normal-approximation 95% CI
ARTIFACTS_DIR = Path("./artifacts/")
FIGURES = Path("report/figures")


from typing import NamedTuple

# easier marshalling
class ReliabilityCurve(NamedTuple):
    mean_pred: np.ndarray
    freq: np.ndarray
    ci: np.ndarray
    counts: np.ndarray

def reliability_points(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> ReliabilityCurve:
    """
    equal-width bins over [0, 1].

    Returns: per-bin arrays: (mean predicted prob, empirical frequency, 95% CI half-width, count),
             NaN for empty bins.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # bin b owns [edges[b], edges[b+1])
    assignments = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    mean_pred, freq, ci, counts = [], [], [], []
    for b in range(n_bins):
        mask = assignments == b
        n_b = int(mask.sum())
        counts.append(n_b)

        if n_b == 0:
            mean_pred.append(np.nan)
            freq.append(np.nan)
            ci.append(np.nan)

        else:
            f = float(y[mask].mean())
            mean_pred.append(float(p[mask].mean()))
            freq.append(f)
            ci.append(Z_95 * np.sqrt(f * (1.0 - f) / n_b))

    return ReliabilityCurve(np.array(mean_pred), np.array(freq), np.array(ci), np.array(counts))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    mean_pred, freq, _, counts = reliability_points(p, y, n_bins)
    return float(np.nansum(counts * np.abs(freq - mean_pred)) / counts.sum())


def plot_reliability(dataset_name, model_name, curves, n_bins, out_path):
    """curves: list of (label, mean_pred, freq, ci). Pure formatting."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")

    for label, curve in curves:
        ok = ~np.isnan(curve.mean_pred)
        sizes = 15 + 70 * np.sqrt(curve.counts[ok] / curve.counts[ok].max())

        ax.errorbar(curve.mean_pred[ok], curve.freq[ok], yerr=curve.ci[ok],
                    # marker="o", markersize=4, capsize=3, linewidth=1.2, label=label)
                    fmt='none', capsize=3, linewidth=1.2)
        ax.scatter(curve.mean_pred[ok], curve.freq[ok], s=sizes,
                   linewidth=0.8, edgecolor='black', zorder=3, label=label)

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title(f"{dataset_name} / {model_name} (M={n_bins})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    plt.close(fig)


def plot_all_diagrams(datasets, n_bins: int = 10) -> None:
    FIGURES.mkdir(exist_ok=True)

    for ds in datasets:
        y_test = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")["y_test"]

        for model_name in ["logreg", "rf"]:
            d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
            pl = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")
            curves = [
                ("raw", reliability_points(d["test_proba"], y_test, n_bins)),
                ("platt", reliability_points(pl["test_calibrated"], y_test, n_bins)),
                ("isotonic", reliability_points(iso["test_calibrated"], y_test, n_bins)),
                ("iso+clip", reliability_points(iso["test_calibrated_clipped"], y_test, n_bins)),
            ]
            plot_reliability(ds.name, model_name, curves, n_bins,
                             FIGURES / f"{ds.name}_{model_name}_reliability.png")

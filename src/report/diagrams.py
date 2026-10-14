from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np

from src.constants import ARTIFACTS_DIR, FIGURES_DIR

Z_95 = 1.96  # 95% CI


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



def plot_all_diagrams(datasets) -> None:
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    variants = ["raw", "platt", "isotonic", "iso+clip"]
    models = ["logreg", "rf", "mlp"]

    for ds in datasets:
        y_test = np.load(ARTIFACTS_DIR / f"{ds.name}_split.npz")["y_test"]
        n_bins = min(10, len(y_test) // 30)

        fig, axes = plt.subplots(
            len(models), len(variants),
            figsize=(3.4 * len(variants), 3.4 * len(models)),
            sharex=True, sharey=True)

        for row, model_name in enumerate(models):
            d = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}.npz")
            pl = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            probs = {
                "raw":      d["test_proba"],
                "platt":    pl["test_calibrated"],
                "isotonic": iso["test_calibrated"],
                "iso+clip": iso["test_calibrated_clipped"],
            }

            for col, vname in enumerate(variants):
                ax = axes[row, col]
                ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

                curve = reliability_points(probs[vname], y_test, n_bins)
                ok = ~np.isnan(curve.mean_pred)

                y_lo = np.minimum(curve.ci[ok], curve.freq[ok])
                y_hi = np.minimum(curve.ci[ok], 1.0 - curve.freq[ok])

                sizes = 15 + 70 * np.sqrt(curve.counts[ok] / curve.counts[ok].max())

                ax.errorbar(curve.mean_pred[ok], curve.freq[ok], yerr=[y_lo, y_hi],
                            fmt="none", capsize=2, linewidth=1, color="C0")
                ax.scatter(curve.mean_pred[ok], curve.freq[ok], s=sizes,
                           color="C0", edgecolor="black", linewidth=0.7, zorder=3)

                ax.set_xlim(0, 1)
                ax.set_ylim(-0.03, 1.03)

                if row == 0:
                    ax.set_title(vname.title(), fontsize=10)

                if col == 0:
                    ax.set_ylabel(f"{model_name.upper()}\nEmpirical Frequency", fontsize=9)

                if row == len(models) - 1:
                    ax.set_xlabel("Mean Predicted Probability", fontsize=9)

        fig.suptitle(f"{ds.name.replace("_", " ").title()} (M={n_bins})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{ds.name}_reliability.png", dpi=300)
        plt.close(fig)

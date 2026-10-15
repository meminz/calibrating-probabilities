# fully vibed
from pathlib import Path
import numpy as np

from src.pipeline import ARTIFACTS_DIR, MODELS

REPORT_TABLES = Path("./report/tables")

# ---------- Pretty-print mappings for LaTeX tables --------------------
MODEL_NAMES = {"logreg": "LogReg", "rf": "RF", "mlp": "MLP"}
VARIANT_NAMES = {
    "raw": "Raw",
    "platt": "Platt",
    "isotonic": "Iso",
    "iso+clip": "Iso+Clip",
}
DATASET_NAMES = {
    "spambase": "Spambase",
    "adult": "Adult",
    "adult_sub": "Adult (Sub)",
}


def _pretty_model(name: str) -> str:
    return MODEL_NAMES.get(name, name.title())


def _pretty_variant(name: str) -> str:
    return VARIANT_NAMES.get(name, name.title())


def _pretty_dataset(name: str) -> str:
    return DATASET_NAMES.get(name, name.title())


def export_results_latex(rows: list[dict], out_dir: Path = REPORT_TABLES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "\\centering\n"
        "\\small\\sffamily\n"
        "\\setlength{\\tabcolsep}{8pt}\n"
        "\\begin{tabular}{ll *{6}{c}}\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{Variant} & "
        "\\multicolumn{2}{c}{\\textbf{Accuracy}} & "
        "\\multicolumn{2}{c}{\\textbf{Log-Loss}} & "
        "\\multicolumn{2}{c}{\\textbf{Brier Score}} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "& & OOF & Test & OOF & Test & OOF & Test \\\\\n"
        "\\midrule\n"
    )

    for ds_name in dict.fromkeys(r["dataset"] for r in rows):
        lines = [header]
        current_model = None
        first_row_of_model = True

        for r in rows:
            if r["dataset"] != ds_name:
                continue

            # Insert a horizontal rule between model groups
            if r["model"] != current_model:
                if current_model is not None:
                    lines.append("\\midrule")
                current_model = r["model"]
                first_row_of_model = True

            model_pretty = _pretty_model(r["model"])
            variant_pretty = _pretty_variant(r["variant"])

            # Only print the model name on the first row of its group
            model_cell = model_pretty if first_row_of_model else ""
            first_row_of_model = False

            lines.append(
                f"{model_cell} & {variant_pretty} & "
                f"{r['acc_oof']:.3f} & {r['acc_test']:.3f} & "
                f"{r['ll_oof']:.4f} & {r['ll_test']:.4f} & "
                f"{r['brier_oof']:.4f} & {r['brier_test']:.4f} \\\\"
            )

        lines.append("\\bottomrule\n\\end{tabular}")
        (out_dir / f"results_{ds_name}.tex").write_text("\n".join(lines))


def export_params_latex(datasets, out_dir: Path = REPORT_TABLES) -> None:
    header = (
        "\\centering\n"
        "\\small\\sffamily\n"
        "\\setlength{\\tabcolsep}{8pt}\n"
        "\\begin{tabular}{ll cc c}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Model} & "
        "\\multicolumn{2}{c}{\\textbf{Platt Scaling}} & \\textbf{Isotonic} \\\\\n"
        "\\cmidrule(lr){3-4}\n"
        "& & $A$ & $B$ & Blocks \\\\\n"
        "\\midrule\n"
    )
    lines = [header]
    current_ds = None
    first_row_of_ds = True

    for ds in datasets:
        if current_ds is not None and ds.name != current_ds:
            lines.append("\\midrule")
            first_row_of_ds = True
        current_ds = ds.name

        ds_pretty = _pretty_dataset(ds.name)

        for model_name in MODELS:
            pl = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            ds_cell = ds_pretty if first_row_of_ds else ""
            first_row_of_ds = False

            model_pretty = _pretty_model(model_name)

            # Wrap B in math mode so the minus sign renders consistently
            lines.append(
                f"{ds_cell} & {model_pretty} & "
                f"{pl['A']:.3f} & ${pl['B']:.3f}$ & "
                f"{int(iso['n_blocks'])} \\\\"
            )

    lines.append("\\bottomrule\n\\end{tabular}")
    (out_dir / "calibrator_params.tex").write_text("\n".join(lines))

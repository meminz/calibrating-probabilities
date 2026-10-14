from pathlib import Path
import numpy as np

from src.pipeline import ARTIFACTS_DIR, MODELS

REPORT_TABLES = Path("./report/tables")

def export_results_latex(rows: list[dict], out_dir: Path = REPORT_TABLES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "model & variant & acc\\_oof & acc\\_test & ll\\_test & br\\_oof & br\\_test \\\\\n"
        "\\midrule"
    )

    for ds_name in dict.fromkeys(r["dataset"] for r in rows):
        lines, current_model = [header], None

        for r in rows:
            if r["dataset"] != ds_name:
                continue

            if r["model"] != current_model:          # visual grouping per model
                current_model = r["model"]
                lines.append("\\addlinespace[2pt]")

            lines.append(
                f"{r['model']} & {r['variant']} & {r['acc_oof']:.3f} & "
                f"{r['acc_test']:.3f} & {r['ll_test']:.4f} & "
                f"{r['brier_oof']:.4f} & {r['brier_test']:.4f} \\\\"
            )

        lines.append("\\bottomrule\n\\end{tabular}")
        (out_dir / f"results_{ds_name}.tex").write_text("\n".join(lines))


def export_params_latex(datasets, out_dir: Path = REPORT_TABLES) -> None:
    lines = [
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "dataset & model & $A$ & $B$ & blocks \\\\\n\\midrule",
    ]

    for ds in datasets:
        for model_name in MODELS:
            pl = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_platt.npz")
            iso = np.load(ARTIFACTS_DIR / f"{ds.name}_{model_name}_isotonic.npz")

            lines.append(f"{ds.name} & {model_name} & {pl['A']:.3f} & {pl['B']:.3f}"
                         f" & {int(iso['n_blocks'])} \\\\")

    lines.append("\\bottomrule\n\\end{tabular}")
    (out_dir / "calibrator_params.tex").write_text("\n".join(lines))

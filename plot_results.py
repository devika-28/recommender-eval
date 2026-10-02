"""Create plots from the Python LensKit evaluation CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "build" / "eval-results.csv"
PLOT_DIR = ROOT / "build" / "plots"


def best_configuration(data: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    summary = (
        data.groupby(["Algorithm", "NNbrs"], dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0)
    summary = summary.dropna(subset=["mean"]).sort_values(
        ["Algorithm", "mean"], ascending=[True, ascending]
    )
    return summary.drop_duplicates("Algorithm", keep="first")


def plot_best(data: pd.DataFrame, metric: str, ascending: bool, title: str, filename: str) -> None:
    summary = best_configuration(data, metric, ascending)
    labels = [
        f"{row.Algorithm} (k={int(row.NNbrs)})" if pd.notna(row.NNbrs) else row.Algorithm
        for row in summary.itertuples(index=False)
    ]
    values = summary["mean"].to_list()
    errors = summary["std"].to_list()
    order = list(range(len(summary)))[::-1]

    fig, axis = plt.subplots(figsize=(10, max(4, 0.48 * len(labels))))
    axis.barh(
        [labels[index] for index in order],
        [values[index] for index in order],
        xerr=[errors[index] for index in order],
        color="#4472C4",
        capsize=3,
    )
    axis.set_xlabel(metric.replace(".", " "))
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_neighborhoods(data: pd.DataFrame) -> None:
    selected = ["UserUserCosine", "ItemItem", "TagContent", "TagContentNorm"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for algorithm in selected:
        rows = data.loc[data["Algorithm"] == algorithm]
        summary = rows.groupby("NNbrs")["TopN.nDCG"].agg(["mean", "std"]).dropna()
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            label=algorithm,
        )
    axis.set_xlabel("Neighborhood size")
    axis.set_ylabel("Top-N nDCG @ 10")
    axis.set_title("Top-N accuracy by neighborhood size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "topn-ndcg-by-neighborhood.png", dpi=160)
    plt.close(figure)


def main() -> None:
    if not RESULTS.is_file():
        raise FileNotFoundError(
            f"Missing {RESULTS}. Run python_eval.py first to create evaluation results."
        )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULTS)
    plot_best(
        data,
        "RMSE.ByUser",
        True,
        "Best RMSE by algorithm (mean across folds; lower is better)",
        "rmse-by-algorithm.png",
    )
    plot_best(
        data,
        "TopN.nDCG",
        False,
        "Best Top-N nDCG @ 10 by algorithm (mean across folds; higher is better)",
        "topn-ndcg-by-algorithm.png",
    )
    plot_neighborhoods(data)
    for image in sorted(PLOT_DIR.glob("*.png")):
        print(image)


if __name__ == "__main__":
    main()

from pathlib import PosixPath

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 18
LINEWIDTH = 5
MARKERSIZE = 18

PELETTE = {
    "MIPS": "tab:blue",
    "MIPS (w/true ROIPS)": "tab:pink",
    "MIPS (w/heuristic ROIPS)": "tab:gray",
    "FM-IPS (DM)": "tab:orange",
}
LINESTYLE = {
    "MIPS": "",
    "MIPS (w/true ROIPS)": "",
    "MIPS (w/heuristic ROIPS)": "",
    "FM-IPS (DM)": "",
}


def visualize_mean_squared_error(result_df: DataFrame, xlabel: str, xscale: str, img_path: PosixPath) -> None:
    estimators = result_df["estimator"].unique()
    palettes = {estimator: PELETTE[estimator] for estimator in estimators}
    linestyles = {estimator: LINESTYLE[estimator] for estimator in estimators}

    xvalue = result_df["x"].unique()
    xvalue_labels = list(map(str, xvalue))

    for yscale in ["linear", "log"]:
        _visualize_mse_bias_variance(
            result_df=result_df,
            xlabel=xlabel,
            xscale=xscale,
            xvalue=xvalue,
            xvalue_labels=xvalue_labels,
            yscale=yscale,
            palettes=palettes,
            linestyles=linestyles,
            img_path=img_path / f"{yscale}_mse.png",
        )


def _visualize_mse_bias_variance(
    result_df: DataFrame,
    xlabel: str,
    xscale: str,
    xvalue: list,
    xvalue_labels: list,
    yscale: str,
    palettes: dict,
    linestyles: dict,
    img_path: PosixPath,
) -> None:
    plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    title = ["MSE", "Squared Bias", "Variance"]
    y = ["se", "bias", "variance"]

    ylims = []
    for i, (ax_, title_, y_) in enumerate(zip(axes, title, y)):
        sns.lineplot(
            data=result_df,
            x="x",
            y=y_,
            hue="estimator",
            style="estimator",
            ci=95 if i == 0 else None,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            ax=ax_,
            palette=palettes,
            dashes=linestyles,
            markers=True,
            legend="full" if i == 0 else False,
        )
        if i == 0:
            handles, labels = ax_.get_legend_handles_labels()
            ax_.legend_.remove()
            for handle in handles:
                handle.set_linewidth(LINEWIDTH)
                handle.set_markersize(MARKERSIZE)

        # title
        ax_.set_title(f"{title_}", fontsize=TITLE_FONTSIZE)
        # xaxis
        if i == 1:
            ax_.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        else:
            ax_.set_xlabel("")

        ax_.set_xscale(xscale)
        ax_.set_xticks(xvalue, xvalue_labels, fontsize=TICK_FONTSIZE)
        ax_.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # yaxis
        ax_.set_yscale(yscale)
        ax_.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax_.set_ylabel("")

        if i == 0:
            ylims = ax_.get_ylim()
            ylims = (0.0, ylims[1])

    if yscale == "linear":
        for ax_ in axes:
            ax_.set_ylim(ylims)

    fig.legend(
        handles,
        labels,
        fontsize=LABEL_FONTSIZE,
        ncol=len(palettes),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
    )

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()

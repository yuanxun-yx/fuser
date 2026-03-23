from pathlib import Path
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from typing import Any

from progress import ProgressReporter


def default_title(group: tuple[Any, ...]) -> str:
    return "_".join([str(x) for x in group])


def plot(
    df: pl.DataFrame,
    save_path: str | Path,
    *,
    fig_cols: tuple[str, ...],
    x_col: str,
    y_col: str,
    hue_col: str,
    min_sample_n: int,
    format: str = "png",
    progress_reporter: ProgressReporter | None = None,
):
    save_path = Path(save_path)

    hue_vals = df[hue_col].unique()
    if hue_vals.len() != 2:
        raise ValueError(
            "column selected for legend in figure should have only 2 groups for hypothesis testing"
        )

    n_df = (
        df.group_by(fig_cols + (x_col, hue_col))
        .agg(pl.count())
        .pivot(hue_col, values=pl.count().meta.output_name())
    )
    n_df = n_df.filter(
        (pl.col(hue_vals[0]) >= min_sample_n) & (pl.col(hue_vals[1]) >= min_sample_n)
    )
    df = df.join(n_df, on=fig_cols + (x_col,), how="semi")

    save_path.mkdir(parents=True, exist_ok=True)

    if progress_reporter is not None:
        n = df.select(fig_cols).n_unique()
        progress_reporter.start(n)

    for fig_group, fig_df in df.group_by(fig_cols):
        title = default_title(fig_group)

        fig, ax = plt.subplots()

        plot_params = dict(data=fig_df, x=x_col, y=y_col, hue=hue_col, ax=ax)

        sns.boxplot(
            legend=False,
            # pass to ax.boxplot
            showfliers=False,
            boxprops=dict(facecolor="none"),
            **plot_params,
        )

        # scatter
        sns.stripplot(jitter=True, dodge=True, **plot_params)  # horizontal jitter

        # significance annotation
        pairs = [tuple((x, h) for h in hue_vals) for x in fig_df[x_col].unique()]
        # statannotations only compatible with pandas (utils.get_x_values())
        plot_params["data"] = fig_df.to_pandas()
        annotator = Annotator(pairs=pairs, **plot_params)
        annotator.configure(
            test="t-test_welch", text_format="star", loc="outside", verbose=False
        )
        annotator.apply_and_annotate()

        ax.set(title=title, xlabel=None, ylabel=None)
        fig.savefig((save_path / title.replace("/", "")).with_suffix(f".{format}"))
        plt.close(fig)

        if progress_reporter is not None:
            progress_reporter.advance()

from pathlib import Path
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from typing import Any

from progress import ProgressReporter


def group_to_title(group: tuple[Any, ...]) -> str:
    return "_".join([str(x) for x in group])


def plot(
    df: pl.DataFrame,
    save_path: str | Path,
    *,
    fig_cols: tuple[str, ...],
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    min_sample_n: int,
    format: str = "png",
    progress_reporter: ProgressReporter | None = None,
):
    save_path = Path(save_path)

    if hue_col is None:
        test_col = x_col
        test_var = "x"
        count_group = (*fig_cols, x_col)
    else:
        test_col = hue_col
        test_var = "hue"
        count_group = (*fig_cols, x_col, hue_col)

    test_vals = df[test_col].unique()
    if test_vals.len() != 2:
        raise ValueError(
            f"column selected for {test_var} in figure should have only 2 groups for hypothesis testing"
        )

    n_df = (
        df.group_by(count_group)
        .agg(pl.count())
        .pivot(test_col, values=pl.count().meta.output_name())
    )
    n_df = n_df.filter(
        (pl.col(test_vals[0]) >= min_sample_n) & (pl.col(test_vals[1]) >= min_sample_n)
    )
    df = df.join(n_df, on=count_group[:-1], how="semi")

    save_path.mkdir(parents=True, exist_ok=True)

    if progress_reporter is not None:
        n = df.select(fig_cols).n_unique()
        progress_reporter.start(n)

    for fig_group, fig_df in df.group_by(fig_cols):
        title = group_to_title(fig_group)

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
        if hue_col is None:
            pairs = [tuple(test_vals)]
        else:
            pairs = [tuple((x, h) for h in test_vals) for x in fig_df[x_col].unique()]
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

import polars as pl


def check_pk(
    df: pl.DataFrame, cols: str | list[str] | tuple[str, ...], *, max_keys: int = 5
) -> None:
    dup_keys = (
        df.group_by(cols)
        .len()
        .filter(pl.col(pl.len().meta.output_name()) > 1)
        .sort(pl.len().meta.output_name(), descending=True)
    )
    if dup_keys.is_empty():
        return
    sample = dup_keys.head(max_keys)
    raise ValueError(f"{len(dup_keys)} non unique values in column {cols}:\n{sample}")

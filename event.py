import numpy as np
from pathlib import Path
import polars as pl


def read_events(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.with_columns(
        pl.col("times").str.split(" ").list.eval(pl.element().cast(pl.UInt16))
    )
    return df


EVENT_NAME = "event"
NON_EVENT_NAME = "non-event"


def get_event_df(
    events: np.ndarray,
    total_time: float,
    hemodynamic_lag: float,
    max_event_n: int,
    min_event_time: float,
    max_event_time: float,
    post_event_exclusion_window: float,
    include_start_for_non_event: bool = False,
) -> tuple[pl.DataFrame, float]:
    def get_df(events: np.ndarray, type: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "onset": events[:, 0],
                "duration": events[:, 1],
                "trial_type": [type] * events.shape[0],
            }
        )

    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError(f"epochs should have shape (n,2), got {events.shape}")

    events = events.astype(np.float32)

    events += hemodynamic_lag
    event_n = events.shape[0]
    non_event_epochs = np.empty((event_n + 1, 2), dtype=events.dtype)
    non_event_epochs[0, 0] = 0.0
    non_event_epochs[-1, 1] = total_time
    non_event_epochs[1:, 0] = events[:, 1] + post_event_exclusion_window
    non_event_epochs[:-1, 1] = events[:, 0]
    if max_event_n >= event_n:
        max_time = total_time
    else:
        max_time = events[max_event_n, 0]
        events = events[:max_event_n, :]
        non_event_epochs = non_event_epochs[: max_event_n + 1, :]
    events[:, 1] -= events[:, 0]
    events = events[events[:, 1] >= min_event_time]
    events[events[:, 1] > max_event_time, 1] = max_event_time
    event_df = get_df(events, EVENT_NAME)

    non_event_epochs = non_event_epochs[int(include_start_for_non_event) :, :]
    non_event_epochs[:, 1] -= non_event_epochs[:, 0]
    non_event_df = get_df(non_event_epochs, NON_EVENT_NAME)

    return pl.concat([event_df, non_event_df]), max_time

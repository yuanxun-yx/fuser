import numpy as np


def make_event(
    intervals: np.ndarray,
    time: np.ndarray,
    *,
    hemodynamic_lag: float,
) -> np.ndarray:
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"intervals shape should be (N, 2), got {intervals.shape}")
    intervals = intervals + hemodynamic_lag
    event = (
        (time[..., None] >= intervals[:, 0]) & (time[..., None] <= intervals[:, 1])
    ).any(axis=-1)
    return event

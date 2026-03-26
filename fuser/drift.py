import numpy as np


def cosine_drift(
    n_time: int,
    dt: float,
    *,
    high_pass: float,
) -> np.ndarray:
    if n_time <= 0:
        raise ValueError("n_time must be greater than 0")
    if dt <= 0:
        raise ValueError("dt must be greater than 0")
    if high_pass < 0:
        raise ValueError("high_pass cannot be negative")

    total_time = n_time * dt
    n_basis = np.floor(2 * total_time * high_pass)
    ks = (np.arange(n_basis) + 1)[None, :]
    n = np.arange(n_time)[:, None]
    drift = np.cos(np.pi * (n + 0.5) * ks / n_time)
    return drift


def make_drift(
    time: np.ndarray,
    *,
    model: str,
    high_pass: float,
) -> np.ndarray:
    diff = np.diff(time)
    if not np.all(diff > 0):
        raise ValueError("time must be monotonically increasing")
    if model == "cosine":
        dt = diff.mean()
        if not np.allclose(diff, dt):
            raise ValueError("dt is not constant")
        return cosine_drift(time.size, dt, high_pass=high_pass)
    else:
        raise ValueError(f"unknown model: {model}")

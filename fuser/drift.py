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

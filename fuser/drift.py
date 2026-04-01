import numpy as np


def make_drift(
    time: np.ndarray,
    *,
    model: str,
    high_pass: float,
) -> np.ndarray:
    if model == "cosine":
        if high_pass < 0:
            raise ValueError("high_pass cannot be negative")

        time = time - time.min()
        T = time.max()
        n_basis = np.floor(2 * T * high_pass)
        ks = (np.arange(n_basis) + 1)[None, :]
        t = time[:, None]
        drift = np.cos(np.pi * ks * t / T)
        return drift
    else:
        raise ValueError(f"unknown model: {model}")

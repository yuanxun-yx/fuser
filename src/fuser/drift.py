import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftConfig:
    model: str
    high_pass: float | None = None
    order: int | None = None


def make_drift(
    time: np.ndarray,
    *,
    model: str,
    high_pass: float | None = None,
    order: int | None = None,
) -> np.ndarray:
    if model == "cosine":
        if high_pass is None or high_pass < 0:
            raise ValueError("high_pass cannot be negative")

        time = time - time.min()
        T = time.max()
        n_basis = np.floor(2 * T * high_pass)
        ks = (np.arange(n_basis) + 1)[None, :]
        t = time[:, None]
        drift = np.cos(np.pi * ks * t / T)
        return drift
    elif model == "polynomial":
        if order is None or order < 0:
            raise ValueError("order cannot be negative")

        time = time - time.min()
        T = time.max()
        t = (time / T)[:, None]

        # constant term not included because intercept is included in GLM
        ks = np.arange(1, order + 1)[None, :]
        drift = t**ks
        return drift
    else:
        raise ValueError(f"unknown model: {model}")

import numpy as np


def mad_outlier_mask(
    x, thresh: float, axis: int | tuple[int] | None = None
) -> np.ndarray:
    med = np.median(x, axis=axis)
    mad = np.median(x - med, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = 0.6744897501960817 * (x - med) / mad
    bad = (np.abs(z) > thresh) & (mad != 0)
    return bad


def detect_global_outliers(data: np.ndarray, thresh: float = 5.0) -> np.ndarray:
    global_signal = data.mean(axis=(-3, -2, -1))
    return mad_outlier_mask(global_signal, axis=0, thresh=thresh)


def detect_motion_outliers(motion: np.ndarray, thresh: float = 5.0) -> np.ndarray:
    n = motion.shape[-1]
    if n != 3:
        raise ValueError(f"last dimension of motion should be 3, got {n}")
    s = np.linalg.norm(motion, axis=-1)
    return mad_outlier_mask(s, thresh=thresh)


def detect_frame_correlation_drop(data: np.ndarray, thresh: float = 0.8) -> np.ndarray:
    x = data.reshape(*data.shape[:2], -1)
    x -= x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        x /= norm
    corr = np.sum(x[1:, ...] * x[:-1, ...], axis=-1)
    bad = np.zeros(data.shape[:2], dtype=bool)
    nonzero = norm.squeeze(-1) != 0
    bad[1:] = (corr < thresh) & nonzero[1:, ...] & nonzero[:-1, ...]
    return bad

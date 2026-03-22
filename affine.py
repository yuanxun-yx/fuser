import numpy as np


def check_valid_transform(t: np.ndarray, *, batch_shape: tuple[int, ...] = tuple()) -> None:
    shape = (*batch_shape, 4, 4)
    if t.shape != shape:
        raise ValueError(f"transform shape should be {shape}, got {t.shape}")

    if not np.all(t[..., 3, :] == [0, 0, 0, 1]):
        raise ValueError(f"last row must be [0,0,0,1], got {t[3]}")

    r = t[..., :3, :3]
    if np.any(np.linalg.matrix_rank(r) < 3):
        raise ValueError(f"rotation determinant transform is singular and not invertible")

import numpy as np
from numpy.linalg import lstsq


def glm_fit(
    fus: np.ndarray,
    regressors: np.ndarray,
    noise_model: str = "ols",
    time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    fus: (scan, pose, x, y, z)
    regressors: (scan, pose, regressor)
    time_mask: (scan, pose)
    beta: (regressor, pose, x, y, z)
    """
    # use events as x to explain fUS as y
    beta = np.empty((regressors.shape[-1], *fus.shape[1:]), dtype=fus.dtype)
    # do it per pose because each pose has different time slices
    # per pose GLM is correct because data is in y not x
    for i in range(fus.shape[1]):
        y = fus[:, i, ...].reshape(fus.shape[0], -1)
        x = regressors[:, i, :]
        if time_mask is not None:
            m = time_mask[:, i]
            y = y[m, :]
            x = x[m, :]
        solution, *_ = lstsq(x, y)
        beta[:, i, ...] = solution.reshape(-1, *fus.shape[-3:])
    return beta

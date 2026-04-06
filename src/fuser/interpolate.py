from scipy.interpolate import interp1d
import numpy as np


def interpolate_pose(
    time: np.ndarray,
    data: np.ndarray,
    *,
    kind="linear",
    fill_value="extrapolate",
) -> np.ndarray:
    """
    per pose interpolation to full time axis
    return: (time, pose, x, y, z)
    time = scan * pose
    """
    time_s = np.sort(time.ravel())
    data_interp = np.empty(
        (data.shape[0] * data.shape[1], *data.shape[1:]), dtype=data.dtype
    )
    for i in range(data.shape[1]):
        f = interp1d(
            time[:, i],
            data[:, i, ...],
            axis=0,
            kind=kind,
            fill_value=fill_value,
        )
        data_interp[:, i, ...] = f(time_s)
    return data_interp

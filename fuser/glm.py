import numpy as np
from numpy.linalg import lstsq

from .registration import motion_correct
from .drift import make_drift
from .event import make_event


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


def run_glm(
    data: np.ndarray,
    time: np.ndarray,
    events: list[np.ndarray],
    *,
    motion: np.ndarray = None,
    hemodynamic_lag: float,
    drift_model: str | None = None,
    high_pass: float | None = None,
    max_time: float | None = None,
) -> np.ndarray:
    """
    we don't use nilearn.first_level directly because data structure is not usual 4d array
    the problem with fUS data is that time and space axes are coupled
    data: (scan, pose, x, y, z)
    time: (scan, pose)
    space: (pose, x, y, z)
    pose determines both time and space, therefore we cannot simply decouple data to (T, N)
    """
    n_events = len(events)

    time_r = time.ravel()
    idx = np.argsort(time_r)
    inverse_idx = np.empty_like(idx)
    inverse_idx[idx] = np.arange(idx.size)
    time_s = time_r[idx]

    if motion is not None:
        if motion.shape[-1] != 3:
            raise ValueError(
                f"size of last dimension of motion should be 3, got {motion.shape[-1]}"
            )
        # left last axis only
        axis = tuple(range(motion.ndim - 1))
        # remove axes (xyz) with all zeros
        motion = motion[..., ~np.all(motion == 0, axis=axis)]
        # z-score motion to prevent ill condition
        motion = (motion - motion.mean(axis=axis)) / motion.std(axis=axis)

    # nuisance
    # per pose global signal
    global_signal = np.mean(data, axis=(-3, -2, -1))
    # center global signal to prevent co-linear with intercept
    global_signal -= global_signal.mean()

    regressors = []
    for e in events:
        e = make_event(e, time_s, hemodynamic_lag=hemodynamic_lag)
        e = e.reshape(-1, 1)
        regressors.append(e)
    if drift_model is not None:
        drift = make_drift(time_s, model=drift_model, high_pass=high_pass)
        regressors.append(drift)
    regressors = np.concatenate(regressors, axis=-1)
    regressors = regressors[inverse_idx, :]
    regressors = regressors.reshape(*time.shape, regressors.shape[-1])

    regressors = [regressors, global_signal[..., None], np.ones((*time.shape, 1))]
    if motion is not None:
        regressors.append(motion)
    regressors = np.concatenate(regressors, axis=-1)

    if max_time is None:
        mask = None
    else:
        mask = time <= max_time

    beta = glm_fit(
        fus=data,
        regressors=regressors,
        time_mask=mask,
    )[:n_events, ...]

    return beta

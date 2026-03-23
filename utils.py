import numpy as np


def bincount_axes(
    x: np.ndarray,
    /,
    axis: int | tuple[int, ...] | None = None,
    weights: np.ndarray = None,
) -> np.ndarray:
    """
    apply numpy.bincount on given axes of x
    """
    if weights is not None:
        weights = np.broadcast_to(weights, x.shape)

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    # move flattened axes to the end
    dest = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, dest)

    batch_shape = x.shape[: -len(axis)]
    reduce_size = np.prod(x.shape[-len(axis) :])

    x = x.reshape(-1, reduce_size)

    k = x.max() + 1
    b = x.shape[0]

    offset = np.arange(b)[:, None] * k
    x = (x + offset).ravel()

    # apply same flatten to weights
    if weights is not None:
        weights = np.moveaxis(weights, axis, dest)
        weights = weights.reshape(-1, reduce_size)
        weights = weights.ravel()

    counts = np.bincount(x, weights=weights, minlength=b * k)

    counts = counts.reshape(*batch_shape, k)

    return counts

import numpy as np

from .utils import bincount_axes


def aggregate_to_roi(
    data: np.ndarray,
    *,
    annotation: np.ndarray,
    mask: np.ndarray,
    roi_ids: list[list[int]],
    thresh: float,
) -> np.ndarray:
    """
    data: (..., pose, x, y, z)
    annotation, mask: (pose, x, y, z)
    roi_aggregate: (..., roi)
    """

    if data.ndim < 4:
        raise ValueError(f"data shape {data.shape} must have at least 4 dimensions (..., pose, x, y, z)")

    shape = data.shape[-4:]
    if annotation.shape != shape:
        raise ValueError(
            f"annotation shape {annotation.shape} does not match data shape {shape}"
        )
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match data shape {shape}")

    roi_aggregate = np.full(
        (*data.shape[:-4], len(roi_ids)), fill_value=np.nan, dtype=data.dtype
    )

    # convert non-consecutive region ids to 0, 1, 2, ...
    ids, inverse = np.unique(annotation, return_inverse=True)
    # (roi,)
    region_voxel_count = bincount_axes(inverse)
    region_valid_voxel_count = bincount_axes(inverse, weights=mask)

    inverse_b = np.broadcast_to(inverse, data.shape)
    # (..., roi)
    region_valid_value_sum = bincount_axes(
        inverse_b, axis=(-4, -3, -2, -1), weights=mask * data
    )

    for i, subtree in enumerate(roi_ids):
        m = np.isin(ids, subtree)
        roi_count = region_voxel_count[m].sum()
        if roi_count == 0:
            continue
        valid_ratio = region_valid_voxel_count[m].sum() / roi_count
        if valid_ratio < thresh:
            continue
        roi_aggregate[..., i] = (
            region_valid_value_sum[:, m].sum(axis=1) / region_valid_voxel_count[m].sum()
        )

    return roi_aggregate

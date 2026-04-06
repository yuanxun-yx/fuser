import numpy as np
from collections.abc import Collection

from .array import bincount_axes, sum_by_membership, shared_axes


class RoiAggregator:
    def __init__(
        self,
        annotation: np.ndarray,
        mask: np.ndarray,
        roi_ids: Collection[Collection[int]],
        *,
        thresh: float,
    ):
        if annotation.shape != mask.shape:
            raise ValueError(
                f"annotation shape {annotation.shape} does not match mask shape {mask.shape}"
            )
        self._annotation = annotation
        self._mask = mask

        # convert non-consecutive region ids to 0, 1, 2, ...
        self._ids, self._inverse = np.unique(annotation, return_inverse=True)

        self._id_voxel_count = bincount_axes(self._inverse)
        self._id_valid_voxel_count = bincount_axes(self._inverse, weights=self._mask)

        self._roi_id_mask = np.empty((len(roi_ids), len(self._ids)), dtype=bool)
        for i, subtree in enumerate(roi_ids):
            self._roi_id_mask[i, :] = np.isin(self._ids, subtree)

        roi_voxel_count = sum_by_membership(self._id_voxel_count, self._roi_id_mask)
        self._roi_masked_voxel_count = sum_by_membership(
            self._id_valid_voxel_count, self._roi_id_mask
        )
        with np.errstate(invalid="ignore"):
            roi_masked_ratio = self._roi_masked_voxel_count / roi_voxel_count

        self._valid_roi_mask = roi_masked_ratio >= thresh

    @property
    def valid_roi_mask(self) -> np.ndarray:
        return self._valid_roi_mask

    def aggregate(self, data: np.ndarray) -> np.ndarray:

        inverse_b = np.broadcast_to(self._inverse, data.shape)
        axis = shared_axes(self._inverse.shape, data.shape)
        id_masked_data_sum = bincount_axes(
            inverse_b, axis=axis, weights=self._mask * data
        )
        roi_masked_data_sum = sum_by_membership(id_masked_data_sum, self._roi_id_mask)
        with np.errstate(invalid="ignore"):
            roi_masked_data_mean = roi_masked_data_sum / self._roi_masked_voxel_count
        return roi_masked_data_mean[..., self._valid_roi_mask]

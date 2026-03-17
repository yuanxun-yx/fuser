from typing import Any
import polars as pl
import numpy as np
from numpy.linalg import inv
from scipy.linalg import lstsq
from scipy.ndimage import affine_transform, label, binary_fill_holes, binary_closing
from scipy.signal import detrend, convolve
from nilearn.glm.first_level import spm_hrf, glover_hrf

from dataset import Dataset

# in um
BRAIN_ORIGIN_CCFv3_COORD_POST_INF_ML = (5600, 5500, 5700)

# Iconeous "brain" coordinate
# direction: left, anterior, superior (dorsal)
# origin: approx 4mm below Bregma
# CCFv3 axis order: posterior, inferior, right
# 'space' in header is incorrect
# source: https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas
BRAIN_TO_ANNOTATION = np.zeros((4, 4))
BRAIN_TO_ANNOTATION[3, 3] = 1
BRAIN_TO_ANNOTATION[0, 1] = -1
BRAIN_TO_ANNOTATION[1, 2] = -1
BRAIN_TO_ANNOTATION[2, 0] = -1
BRAIN_TO_ANNOTATION[:3, 3] = BRAIN_ORIGIN_CCFv3_COORD_POST_INF_ML
# magic number from Iconeous: 4mm (4000 um)
BRAIN_TO_ANNOTATION[:3, :3] *= 4e3


def bincount_axes(
        x: np.ndarray,
        /,
        axis: int | tuple[int, ...] | None = None,
        weights: np.ndarray = None,
) -> np.ndarray:
    """
    apply numpy.bincount on given axes of x
    """
    if weights is not None and x.shape != weights.shape:
        raise ValueError(f'x {x.shape} and weights {weights.shape} shape do not match')

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    # move flattened axes to the end
    dest = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, dest)

    batch_shape = x.shape[:-len(axis)]
    reduce_size = np.prod(x.shape[-len(axis):])

    x = x.reshape(-1, reduce_size)

    k = x.max().item() + 1
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


def process_fus(
        dataset: Dataset,
        *,
        annotation_header: dict[str, Any],
        annotation_data: np.ndarray,
        voxel_percentile_thresh: float,
        valid_region_voxel_ratio: float,
        fus_delay_s: float,
) -> pl.DataFrame:
    annotation_transform = np.eye(4)
    annotation_transform[:3, :3] = annotation_header['space directions']
    annotation_transform[:3, 3] = annotation_header['space origin']

    dfs = []
    # vectorize this in the future, size of session is ~200MB
    for session in dataset:
        fus_scan = session.fus_scan

        n_block_repeat = fus_scan.data.shape[2]
        if n_block_repeat != 1:
            raise NotImplementedError(f'block repeat number is {n_block_repeat}, we only handle 1 currently')
        data = fus_scan.data.squeeze(2)

        time = fus_scan.acquisition.time.squeeze(2)
        time += fus_delay_s

        epochs = session.epochs
        # shape: event, scan repeat, pose
        event_mask = np.empty((len(epochs), *data.shape[:2]), dtype=bool)
        for i, v in enumerate(session.epochs.values()):
            event_mask[i] = ((time[..., None] >= v[:, 0]) & (time[..., None] <= v[:, 1])).any(axis=-1)
        # haemodynamic response
        diff = np.diff(time, axis=0)
        dt = diff.mean()
        if not np.allclose(dt, diff):
            raise ValueError(f"dt not identical for session {fus_scan.metadata.file_id.hex()}")
        hrf = glover_hrf(dt)
        kernel = np.array([.25, .5, .25])
        # shape: event + 1, scan repeat, pose
        design = np.empty((event_mask.shape[0] + 1, *event_mask.shape[1:]), dtype=hrf.dtype)
        # intercept
        design[-1] = 1
        for i in range(event_mask.shape[0]):
            for j in range(event_mask.shape[-1]):
                design[i, :, j] = convolve(event_mask[i, :, j], kernel, mode='same')

        # each pose + 3d correspond to one voxel in fixed space, not 3d alone
        # because probe is moving, think as 4d coordinate
        # check if each 4d point is valid
        data_avg_scans = data.mean(axis=0)
        threshold = np.percentile(data_avg_scans, voxel_percentile_thresh)
        # shape: pose, x, y, z
        mask = data_avg_scans > threshold
        # morphology process each 3d mask
        for i in range(mask.shape[0]):
            s = mask[i]
            labels, num = label(s)
            sizes = np.bincount(labels.ravel())
            largest_label = np.argmax(sizes[1:]) + 1
            body_mask = labels == largest_label
            # ignore scan direction because it has much smaller size
            for k in range(body_mask.shape[1]):
                s[:, k, :] = binary_closing(binary_fill_holes(body_mask[:, k, :]))

        # do not perform log here to preserve linearity
        data = detrend(data, axis=0)
        # z-score: within scan repeats
        # mean is done by detrend
        std = data.std(axis=0, keepdims=True)
        # avoid 0/0 = NaN
        data = np.where(std > 0, data / std, 0)

        # GLM: to use event to explain fUS
        # shape: pose, scan repeat, event + 1
        x = design.T
        # shape: pose, scan repeat, voxels
        y = data.reshape(*data.shape[:2], -1).swapaxes(0, 1)
        # shape: pose, event + 1, voxels
        beta, *_ = lstsq(x, y)
        # shape: event, pose, x, y, z
        beta = beta.swapaxes(0, 1)[:-1]
        beta = beta.reshape(*beta.shape[:2], *data.shape[-3:])

        voxels_to_annotation_index = (
                inv(annotation_transform) @ BRAIN_TO_ANNOTATION @ inv(session.brain_to_lab) @
                fus_scan.acquisition.probe_to_lab @ fus_scan.acquisition.voxels_to_probe
        )

        # shape: pose, x, y, z (no scan repeat because only depend on pose)
        voxel_annotations = np.empty(data.shape[1:], dtype=annotation_data.dtype)
        for i in range(data.shape[1]):
            voxel_annotations[i] = affine_transform(
                annotation_data,
                matrix=voxels_to_annotation_index[i],
                output_shape=data.shape[-3:],
                order=0  # nearest neighbor
            )

        # convert non-consecutive region ids to 0, 1, 2, ...
        ids, inverse = np.unique(voxel_annotations, return_inverse=True)
        # remember to ignore the background (0)
        background_index = np.where(ids == 0)[0][0]
        # shape: id count
        region_voxel_count = bincount_axes(inverse)
        region_valid_voxel_count = bincount_axes(inverse, weights=mask)
        # shape: event, id count
        inverse_b = np.broadcast_to(inverse, beta.shape)
        region_valid_beta_sum = bincount_axes(inverse_b, axis=tuple(range(-4, 0)),
                                              weights=mask[None, ...] * beta)

        # shape: id count
        # contains NaN if valid count is 0
        with np.errstate(invalid='ignore'):
            region_valid_ratio = region_valid_voxel_count / region_voxel_count
            region_beta_mean = region_valid_beta_sum / region_valid_voxel_count
        valid_region_mask = region_valid_ratio >= valid_region_voxel_ratio
        # ignore background here
        valid_region_mask[background_index] = False

        beta_mean_mask = ~np.isnan(region_beta_mean) & valid_region_mask[None, ...]

        for i, k in enumerate(epochs.keys()):
            m = beta_mean_mask[i]
            n = m.sum()

            dfs.append(pl.DataFrame({
                'session': [fus_scan.metadata.file_id] * n,
                'subject': [session.subject] * n,
                **{name: [cond] * n for name, cond in zip(dataset.CONDITION_NAMES, session.conditions)},
                'epoch_condition': [k] * n,
                'brain_region_id': ids[m],
                'value': region_beta_mean[i, m],
            }))

    df = pl.concat(dfs)
    return df

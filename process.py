from pathlib import Path
import warnings
import nrrd
import polars as pl
import numpy as np
from numpy.linalg import inv
from scipy.ndimage import affine_transform, label, binary_fill_holes, binary_closing

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


def bincount_axes(x: np.ndarray, /, axis: int | tuple[int, ...], weights: np.ndarray = None) -> np.ndarray:
    """
    apply numpy.bincount on given axes of x
    """
    if weights is not None and x.shape != weights.shape:
        raise ValueError(f'x {x.shape} and weights {weights.shape} shape do not match')

    if isinstance(axis, int):
        axis = (axis,)

    # move flattened axes to the end
    dest = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, dest)

    batch_shape = x.shape[:-len(axis)]
    reduce_size = np.prod(x.shape[-len(axis):]).item()

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
        dataset_path: str | Path,
        *,
        annotation_path: str | Path,
        save_path: str | Path,
        voxel_percentile_thresh: float,
        voxel_norm_percentile: float,
        valid_region_voxel_ratio: float,
        valid_region_pose_ratio: float,

):
    annotation_data, annotation_header = nrrd.read(annotation_path)

    annotation_transform = np.eye(4)
    annotation_transform[:3, :3] = annotation_header['space directions']
    annotation_transform[:3, 3] = annotation_header['space origin']

    dfs = []
    dataset = Dataset(dataset_path)
    for session in dataset:
        fus_scan = session.fus_scan

        n_block_repeat = fus_scan.data.shape[2]
        if n_block_repeat != 1:
            raise NotImplementedError(f'block repeat number is {n_block_repeat}, we only handle 1 currently')
        data = fus_scan.data.squeeze(2)

        time = fus_scan.acquisition.time.squeeze(2)
        event_mask = {}
        for k, v in session.epochs.items():
            event_mask[k] = ((time[..., None] >= v[:, 0]) & (time[..., None] <= v[:, 1])).any(axis=-1)

        threshold = np.percentile(data, voxel_percentile_thresh, axis=(2, 3, 4),
                                  keepdims=True)
        mask = data > threshold
        # morphology process each 3d mask
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                s = mask[i, j]
                labels, num = label(s)
                sizes = np.bincount(labels.ravel())
                largest_label = np.argmax(sizes[1:]) + 1
                body_mask = labels == largest_label
                # ignore scan direction because it has much smaller size
                for k in range(body_mask.shape[1]):
                    s[:, k, :] = binary_closing(binary_fill_holes(body_mask[:, k, :]))

        data = np.log(data)
        per_body_norm = np.percentile(data, voxel_norm_percentile, axis=(2, 3, 4),
                                      keepdims=True)
        data /= per_body_norm

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

        body_3d_axes = (-3, -2, -1)
        # convert non-consecutive region ids to 0, 1, 2, ...
        ids, inverse = np.unique(voxel_annotations, return_inverse=True)
        # remember to ignore the background (0)
        background_index = np.where(ids == 0)[0][0]
        inverse_b = np.broadcast_to(inverse, mask.shape)
        # shape: pose, id count
        region_voxel_count = bincount_axes(inverse, axis=body_3d_axes)
        # shape: repeat, pose, id count
        region_valid_voxel_count = bincount_axes(inverse_b, axis=body_3d_axes, weights=mask)
        region_valid_sum = bincount_axes(inverse_b, axis=body_3d_axes, weights=mask * data)

        # shape: scan repeat, pose, id count
        # contains nan if valid count is 0
        with np.errstate(invalid='ignore'):
            region_valid_mean = region_valid_sum / region_valid_voxel_count
            region_valid_ratio = region_valid_voxel_count / region_voxel_count[None, ...]
        valid_region_mask = region_valid_ratio >= valid_region_voxel_ratio

        # check if each region is stable in certain pose across scan repeats
        region_valid_ratio_at_pose = valid_region_mask.mean(axis=0)
        # shape: (pose, id count)
        valid_pose_region_mask = region_valid_ratio_at_pose >= valid_region_pose_ratio

        for k, v in event_mask.items():
            # slice belong to group & region has enough valid voxels
            m = v[..., None] & valid_region_mask & valid_pose_region_mask[None, ...]
            masked_region_mean = np.where(m, region_valid_mean, np.nan)
            # take mean of each pose first, then mean among poses
            # because the number of each pose in each group is not the same
            # shape: (pose, id count)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                mean_per_pose = np.nanmean(masked_region_mean, axis=0)
                event_region_mean = np.nanmean(mean_per_pose, axis=0)

            region_mask = ~np.isnan(event_region_mean)
            # ignore background here
            region_mask[background_index] = False
            n = region_mask.sum()
            dfs.append(pl.DataFrame({
                'subject': [session.subject] * n,
                **{name: [cond] * n for name, cond in zip(dataset.CONDITION_NAMES, session.conditions)},
                'epoch_condition': [k] * n,
                'brain_region_id': ids[region_mask],
                'value': event_region_mean[region_mask],
            }))

    df = pl.concat(dfs)
    df.write_parquet(save_path)

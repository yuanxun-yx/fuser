from typing import Any
import polars as pl
import numpy as np
from nilearn.glm import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from numpy.linalg import inv, svd
from scipy.ndimage import affine_transform, label, binary_fill_holes, binary_closing, shift
from skimage.registration import phase_cross_correlation

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

EVENT_NAME = 'event'
NON_EVENT_NAME = 'non-event'


def get_event_df(
        epochs: np.ndarray,
        total_time: float,
        hemodynamic_lag: float,
        max_event_n: int,
        min_event_time: float,
        max_event_time: float,
        post_event_exclusion_window: float,
        include_start_for_non_event: bool = False,
) -> tuple[pl.DataFrame, float]:
    def get_df(events: np.ndarray, type: str) -> pl.DataFrame:
        return pl.DataFrame({
            'onset': events[:, 0],
            'duration': events[:, 1],
            'trial_type': [type] * events.shape[0],
        })

    if epochs.ndim != 2 or epochs.shape[1] != 2:
        raise ValueError(f"epochs should have shape (n,2), got {epochs.shape}")

    epochs += hemodynamic_lag
    event_n = epochs.shape[0]
    non_event_epochs = np.empty((event_n + 1, 2), dtype=epochs.dtype)
    non_event_epochs[0, 0] = .0
    non_event_epochs[-1, 1] = total_time
    non_event_epochs[1:, 0] = epochs[:, 1] + post_event_exclusion_window
    non_event_epochs[:-1, 1] = epochs[:, 0]
    if max_event_n >= event_n:
        max_time = total_time
    else:
        max_time = epochs[max_event_n, 0]
        epochs = epochs[:max_event_n, :]
        non_event_epochs = non_event_epochs[:max_event_n + 1, :]
    epochs[:, 1] -= epochs[:, 0]
    epochs = epochs[epochs[:, 1] >= min_event_time]
    epochs[epochs[:, 1] > max_event_time, 1] = max_event_time
    event_df = get_df(epochs, EVENT_NAME)

    non_event_epochs = non_event_epochs[int(include_start_for_non_event):, :]
    non_event_epochs[:, 1] -= non_event_epochs[:, 0]
    non_event_df = get_df(non_event_epochs, NON_EVENT_NAME)

    return pl.concat([event_df, non_event_df]), max_time


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
        hemodynamic_lag: float,
        max_event_n: int,
        min_event_time: float,
        max_event_time: float,
        post_event_exclusion_window: float,
        pca_n_components: int,
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

        # we don't use nilearn.first_level directly because data structure is not usual 4d array
        # the problem with fUS data is that time and space axes are coupled
        # data: (scan, pose, x, y, z)
        # time: (scan, pose)
        # space: (pose, x, y, z)
        # pose determines both time and space, therefore we cannot simply decouple data to (T, N)

        # in session registration
        ref = data.mean(axis=(0, 1))
        motion = np.empty((*data.shape[:2], 3))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                mv = data[i, j]
                sh, *_ = phase_cross_correlation(
                    reference_image=ref,
                    moving_image=mv,
                    upsample_factor=2,
                )
                motion[i, j] = sh
                data[i, j] = shift(mv, sh, order=1)
        motion = motion.reshape(-1, 3)

        # voxel mask: check if each 4d space point is valid
        mean = data.mean(axis=0)
        threshold = np.percentile(mean, voxel_percentile_thresh)
        # (pose, x, y, z)
        mask = mean > threshold
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

        # preprocess: z-score
        mean = mean[None, ...]
        std = np.std(data, axis=0, keepdims=True)
        # avoid 0/0 = NaN
        data = np.where(std > 0, (data - mean) / std, 0)

        # flatten & sort time axes
        time_r = time.ravel()
        idx = np.argsort(time_r)
        time_s = time_r[idx]

        # nuisance
        local_signal = np.mean(data, axis=(-3, -2, -1))
        local_signal = local_signal.reshape(-1, 1)

        data_f = data.reshape(data.shape[0] * data.shape[1], -1)
        u, *_ = svd(data_f, full_matrices=False)
        pca = u[:, :pca_n_components]

        confounds = np.concatenate([motion, local_signal, pca], axis=1)
        confounds = confounds[idx, :]

        # GLM: use events as X to explain fUS as y, not use fUS to predict events

        event_df, max_time = get_event_df(
            epochs=session.epochs,
            total_time=time_s[-1],
            hemodynamic_lag=hemodynamic_lag,
            max_event_n=max_event_n,
            min_event_time=min_event_time,
            max_event_time=max_event_time,
            post_event_exclusion_window=post_event_exclusion_window,
        )

        design = make_first_level_design_matrix(
            frame_times=time_s,
            events=event_df.to_pandas(),
            hrf_model='glover',
            drift_model='cosine',
            high_pass=.01,
            add_regs=confounds
        )
        x = design.values
        x = x.reshape(*time.shape, x.shape[1])

        contrast = np.zeros(x.shape[-1])
        contrast[design.columns.get_loc(EVENT_NAME)] = 1
        contrast[design.columns.get_loc(NON_EVENT_NAME)] = -1

        result = np.empty((2, *data.shape[1:]))
        for i in range(data.shape[1]):
            time_mask = time[:, i] <= max_time
            y = data[:, i, ...].reshape(data.shape[0], -1)
            labels, res = run_glm(Y=y[time_mask, :], X=x[time_mask, i, :], noise_model='ols')
            # con = compute_contrast(labels=labels, regression_result=res, con_val=contrast, stat_type='t')
            result[:, i, ...] = res[0].theta[:2, :].reshape((2, *data.shape[-3:]))

        beta = result

        voxels_to_annotation_index = (
                inv(annotation_transform) @ BRAIN_TO_ANNOTATION @ inv(session.brain_to_lab) @
                fus_scan.acquisition.probe_to_lab @ fus_scan.acquisition.voxels_to_probe
        )

        # (pose, x, y, z) (no scan because only depend on pose)
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
        # (id count,)
        region_voxel_count = bincount_axes(inverse)
        region_valid_voxel_count = bincount_axes(inverse, weights=mask)
        # (event, id count)
        inverse_b = np.broadcast_to(inverse, beta.shape)
        region_valid_beta_sum = bincount_axes(inverse_b, axis=tuple(range(-4, 0)),
                                              weights=mask[None, ...] * beta)

        # (id count)
        # contains NaN if valid count is 0
        with np.errstate(invalid='ignore'):
            region_valid_ratio = region_valid_voxel_count / region_voxel_count
            region_beta_mean = region_valid_beta_sum / region_valid_voxel_count
        valid_region_mask = region_valid_ratio >= valid_region_voxel_ratio
        # ignore background here
        valid_region_mask[background_index] = False

        beta_mean_mask = ~np.isnan(region_beta_mean) & valid_region_mask[None, ...]

        for i, j in enumerate([EVENT_NAME, NON_EVENT_NAME]):
            m = beta_mean_mask[i]
            n = m.sum()

            dfs.append(pl.DataFrame({
                'session': [fus_scan.metadata.file_id] * n,
                'subject': [session.subject] * n,
                **{name: [cond] * n for name, cond in zip(dataset.CONDITION_NAMES, session.conditions)},
                'epoch_condition': [j] * n,
                'brain_region_id': ids[m],
                'value': region_beta_mean[i, m],
            }))

    df = pl.concat(dfs)
    return df

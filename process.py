import polars as pl
import numpy as np
from rich.progress import track
from nilearn.glm import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from numpy.linalg import svd
from scipy.ndimage import label, binary_fill_holes, binary_closing, shift
from skimage.registration import phase_cross_correlation

from dataset import Dataset
from ontology import RoiIds
from registration import transform
from epochs import get_event_df, EVENT_NAME, NON_EVENT_NAME


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

    batch_shape = x.shape[:-len(axis)]
    reduce_size = np.prod(x.shape[-len(axis):])

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


def process_fus(
        dataset: Dataset,
        *,
        roi_ids: RoiIds,
        annotation_data: np.ndarray,
        annotation_transform: np.ndarray,
        voxel_percentile_thresh: float,
        valid_region_voxel_ratio: float,
        hemodynamic_lag: float,
        max_event_n: int,
        min_event_time: float,
        max_event_time: float,
        post_event_exclusion_window: float,
        pca_n_components: int,
        show_progress: bool = True,
) -> pl.DataFrame:
    dfs = []
    # vectorize this in the future, size of session is ~200MB
    for session in track(dataset, description='processing sessions...'):
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

        voxel_annotations = transform(
            annotation_data=annotation_data,
            shape=data.shape[1:],
            annotation_transform=annotation_transform,
            brain_to_lab=session.brain_to_lab,
            probe_to_lab=fus_scan.acquisition.probe_to_lab,
            voxels_to_probe=fus_scan.acquisition.voxels_to_probe,
        )

        # convert non-consecutive region ids to 0, 1, 2, ...
        ids, inverse = np.unique(voxel_annotations, return_inverse=True)
        # (id count,)
        region_voxel_count = bincount_axes(inverse)
        region_valid_voxel_count = bincount_axes(inverse, weights=mask)
        # (event, id count)
        inverse_b = np.broadcast_to(inverse, beta.shape)
        region_valid_value_sum = bincount_axes(inverse_b, axis=tuple(range(-4, 0)), weights=mask * beta)

        for roi, subtree in roi_ids.items():
            m = np.isin(ids, subtree)
            roi_count = region_voxel_count[m].sum()
            if roi_count == 0:
                continue
            valid_ratio = region_valid_voxel_count[m].sum() / roi_count
            if valid_ratio < valid_region_voxel_ratio:
                continue
            valid_value_mean = region_valid_value_sum[:, m].sum(axis=1) / region_valid_voxel_count[m].sum()
            for i, k in enumerate([EVENT_NAME, NON_EVENT_NAME]):
                dfs.append({
                    'session': fus_scan.metadata.file_id,
                    'subject': session.subject,
                    **{name: cond for name, cond in zip(dataset.CONDITION_NAMES, session.conditions)},
                    'epoch_condition': k,
                    'roi': roi,
                    'value': valid_value_mean[i],
                })

    df = pl.DataFrame(dfs)
    return df

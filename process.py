import polars as pl
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from scipy.ndimage import label, binary_fill_holes, binary_closing, shift
from skimage.registration import phase_cross_correlation

from dataset import Dataset
from ontology import RoiIds
from registration import transform
from event import build_event_df
from progress import ProgressReporter
from utils import bincount_axes


def correlation(
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
    event_name: str,
    non_event_name: str,
    progress_reporter: ProgressReporter | None = None,
) -> pl.DataFrame:
    dfs = []

    if progress_reporter is not None:
        progress_reporter.start(len(dataset))
    # vectorize this in the future, size of session is ~200MB
    for session in dataset:
        fus_scan = session.fus_scan

        n_block_repeat = fus_scan.data.shape[2]
        if n_block_repeat != 1:
            raise NotImplementedError(
                f"block repeat number is {n_block_repeat}, we only handle 1 currently"
            )
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
        for j in range(data.shape[0]):
            for k in range(data.shape[1]):
                mv = data[j, k, ...]
                sh, *_ = phase_cross_correlation(
                    reference_image=ref,
                    moving_image=mv,
                    upsample_factor=2,
                )
                motion[j, k, :] = sh
                data[j, k, ...] = shift(mv, sh, order=1)
        motion = motion.reshape(-1, 3)
        # remove axes with all zeros
        motion = motion[:, ~np.all(motion == 0, axis=0)]
        # z-score motion to prevent ill condition
        motion = (motion - motion.mean(axis=0)) / motion.std(axis=0)

        # voxel mask: check if each 4d space point is valid
        mean = data.mean(axis=0)
        threshold = np.percentile(mean, voxel_percentile_thresh)
        # (pose, x, y, z)
        mask = mean > threshold
        # morphology process each 3d mask
        for j in range(mask.shape[0]):
            s = mask[j, ...]
            labels, num = label(s)
            sizes = np.bincount(labels.ravel())
            largest_label = np.argmax(sizes[1:]) + 1
            body_mask = labels == largest_label
            # ignore scan direction because it has much smaller size
            for k in range(body_mask.shape[1]):
                s[:, k, :] = binary_closing(binary_fill_holes(body_mask[:, k, :]))

        # flatten & sort time axes
        time_r = time.ravel()
        idx = np.argsort(time_r)
        time_s = time_r[idx]

        # nuisance
        # per pose global signal
        global_signal_pose = np.mean(data, axis=(-3, -2, -1))
        global_signal_pose = global_signal_pose.reshape(-1, 1)
        # center global signal to prevent co-linear with intercept
        global_signal_pose -= global_signal_pose.mean()

        confounds = np.concatenate([motion, global_signal_pose], axis=1)
        confounds = confounds[idx, :]

        # GLM: use events as X to explain fUS as y, not use fUS to predict events

        event_df, max_time = build_event_df(
            events=session.events,
            total_time=time_s[-1],
            hemodynamic_lag=hemodynamic_lag,
            max_event_n=max_event_n,
            min_event_time=min_event_time,
            max_event_time=max_event_time,
            post_event_exclusion_window=post_event_exclusion_window,
            event_name=event_name,
            non_event_name=non_event_name,
        )

        design = make_first_level_design_matrix(
            frame_times=time_s,
            events=event_df.to_pandas(),
            hrf_model="glover",
            drift_model="cosine",
            high_pass=0.01,
            add_regs=confounds,
        )
        x = design.values
        x = x.reshape(*time.shape, x.shape[1])

        result = np.empty((2, *data.shape[1:]))
        # per pose GLM is correct because data is in y not x
        for j in range(data.shape[1]):
            time_mask = time[:, j] <= max_time
            y = data[:, j, ...].reshape(data.shape[0], -1)
            labels, res = run_glm(
                Y=y[time_mask, :], X=x[time_mask, j, :], noise_model="ols"
            )
            result[:, j, ...] = res[0].theta[:2, :].reshape((2, *data.shape[-3:]))

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
        region_valid_value_sum = bincount_axes(
            inverse_b, axis=tuple(range(-4, 0)), weights=mask * beta
        )

        for roi, subtree in roi_ids.items():
            m = np.isin(ids, subtree)
            roi_count = region_voxel_count[m].sum()
            if roi_count == 0:
                continue
            valid_ratio = region_valid_voxel_count[m].sum() / roi_count
            if valid_ratio < valid_region_voxel_ratio:
                continue
            valid_value_mean = (
                region_valid_value_sum[:, m].sum(axis=1)
                / region_valid_voxel_count[m].sum()
            )
            for j, k in enumerate((event_name, non_event_name)):
                dfs.append(
                    {
                        "session": session.id,
                        "event": k,
                        "roi": roi,
                        "value": valid_value_mean[j],
                    }
                )

        if progress_reporter is not None:
            progress_reporter.advance()

    df = pl.DataFrame(dfs)
    return df

import polars as pl
import numpy as np
from numpy.linalg import lstsq

from fuser import (
    RoiIds,
    register_atlas_to_fus,
    ProgressReporter,
    aggregate_to_roi,
    motion_correct,
    compute_valid_mask,
    make_drift,
    make_event,
    glm_fit,
)

from dataset import Dataset
from event import event_intervals


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
    event_name_all = (event_name, non_event_name)

    if progress_reporter is not None:
        progress_reporter.start(len(dataset))
    # vectorize this in the future, size of session is ~200MB
    for session in dataset:
        fus_scan = session.fus_scan
        data = fus_scan.data
        time = fus_scan.acquisition.time

        # we don't use nilearn.first_level directly because data structure is not usual 4d array
        # the problem with fUS data is that time and space axes are coupled
        # data: (scan, pose, x, y, z)
        # time: (scan, pose)
        # space: (pose, x, y, z)
        # pose determines both time and space, therefore we cannot simply decouple data to (T, N)

        # flatten & sort time axes
        time_r = time.ravel()
        idx = np.argsort(time_r)
        inverse_idx = np.empty_like(idx)
        inverse_idx[idx] = np.arange(idx.size)
        time_s = time_r[idx]

        # in session registration
        data, motion = motion_correct(data)
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

        drift = make_drift(time_s, model="cosine", high_pass=0.005)

        events, non_events, max_time = event_intervals(
            events=session.events,
            total_time=time_s[-1],
            max_event_n=max_event_n,
            min_event_time=min_event_time,
            max_event_time=max_event_time,
            post_event_exclusion_window=post_event_exclusion_window,
        )

        regressors = []
        for e in events, non_events:
            e = make_event(e, time_s, hemodynamic_lag=hemodynamic_lag)
            e = e.reshape(-1, 1)
            regressors.append(e)

        regressors.append(drift)

        regressors = np.concatenate(regressors, axis=-1)
        regressors = regressors[inverse_idx, :]
        regressors = regressors.reshape(*time.shape, regressors.shape[-1])
        regressors = np.concatenate(
            [regressors, motion, global_signal[..., None], np.ones((*time.shape, 1))],
            axis=-1,
        )

        beta = glm_fit(
            fus=data,
            regressors=regressors,
            time_mask=time <= max_time,
        )[:2, ...]

        voxel_annotations = register_atlas_to_fus(
            annotation_data=annotation_data,
            shape=data.shape[1:],
            annotation_transform=annotation_transform,
            brain_to_lab=session.brain_to_lab,
            probe_to_lab=fus_scan.acquisition.probe_to_lab,
            voxels_to_probe=fus_scan.acquisition.voxels_to_probe,
        )

        # voxel mask: check if each 4d space point is valid
        mask = compute_valid_mask(data, thresh=voxel_percentile_thresh)

        roi_aggregate = aggregate_to_roi(
            beta,
            annotation=voxel_annotations,
            mask=mask,
            roi_ids=roi_ids.values(),
            thresh=valid_region_voxel_ratio,
        )

        roi_mask = ~np.isnan(roi_aggregate[0])
        rois = np.array(list(roi_ids.keys()))[roi_mask]
        n = roi_mask.sum()

        for e, arr in zip(event_name_all, roi_aggregate):
            dfs.append(
                pl.DataFrame(
                    {
                        "session": [session.id] * n,
                        "event": [e] * n,
                        "roi": rois,
                        "value": arr[roi_mask],
                    }
                )
            )

        if progress_reporter is not None:
            progress_reporter.advance()

    df = pl.concat(dfs)
    return df

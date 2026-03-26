import polars as pl
import numpy as np

from fuser import (
    RoiIds,
    register_atlas_to_fus,
    ProgressReporter,
    aggregate_to_roi,
    compute_valid_mask,
    run_glm,
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

        events, non_events, max_time = event_intervals(
            events=session.events,
            total_time=time.max(),
            max_event_n=max_event_n,
            min_event_time=min_event_time,
            max_event_time=max_event_time,
            post_event_exclusion_window=post_event_exclusion_window,
        )

        beta = run_glm(
            data=data,
            time=time,
            events=[events, non_events],
            hemodynamic_lag=hemodynamic_lag,
            drift_model="cosine",
            high_pass=0.005,
            max_time=max_time,
        )

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

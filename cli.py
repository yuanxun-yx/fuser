from pathlib import Path
import tomllib
import argparse
import polars as pl
from rich.logging import RichHandler
from rich.progress import Progress, track
import numpy as np
import logging

from fuser import (
    find_roi_ids,
    load_annotation,
    plot,
    register_atlas_to_fus,
    aggregate_to_roi,
    compute_valid_mask,
    run_glm,
)

from dataset import Dataset
from event import event_intervals
from progress import RichProgressReporter
from schema import check_pk

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


def pipeline(config: dict):
    paths = config["paths"]
    dataset_paths = paths["dataset"]

    fus_region_values_path = Path(paths["cache"]["fus_region_values"])
    if not fus_region_values_path.is_file():
        with open(paths["input"]["roi"], "r") as f:
            rois = [l for l in f.read().splitlines() if l]

        roi_ids = find_roi_ids(rois)

        annotation_data, annotation_transform = load_annotation()

        dataset = Dataset(
            root_path=dataset_paths["root"],
            session_path=dataset_paths["session"],
            event_path=dataset_paths["event"],
        )

        glm_params = config["glm"]

        dfs = []

        # vectorize this in the future, size of session is ~200MB
        for session in track(dataset, description="processing raw fUS scans..."):
            fus_scan = session.fus_scan
            data = fus_scan.data
            time = fus_scan.acquisition.time

            events, non_events, max_time = event_intervals(
                events=session.events, total_time=time.max(), **config["event"]
            )

            beta = run_glm(
                data=data,
                time=time,
                events=[events, non_events],
                hemodynamic_lag=glm_params["event"]["hemodynamic_lag"],
                drift_model=glm_params["drift"]["model"],
                high_pass=glm_params["drift"]["high_pass"],
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
            mask = compute_valid_mask(data, thresh=config["mask"]["thresh"])

            roi_aggregate = aggregate_to_roi(
                beta,
                annotation=voxel_annotations,
                mask=mask,
                roi_ids=roi_ids.values(),
                thresh=config["roi"]["thresh"],
            )

            roi_mask = ~np.isnan(roi_aggregate[0])
            rois = np.array(list(roi_ids.keys()))[roi_mask]
            n = roi_mask.sum()

            for e, arr in zip(("social", "non-social"), roi_aggregate):
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

        df = pl.concat(dfs)

        df.write_parquet(fus_region_values_path)
        logger.info(f'processed fus data saved to "{fus_region_values_path}"')
    else:
        logger.info(f'loading processed fus data from "{fus_region_values_path}"')
        df = pl.read_parquet(fus_region_values_path)

    check_pk(df, ("session", "event", "roi"))

    # fetch additional information for sessions
    session = pl.read_csv(dataset_paths["session"])
    check_pk(session, "fus")
    df = df.join(session, left_on="session", right_on="fus", how="left")

    genotype = pl.read_csv(dataset_paths["genotype"])
    check_pk(genotype, "subject")
    df = df.join(genotype, on="subject", how="left")

    plots_path = Path(paths["output"]["plots"])
    with Progress() as progress:
        reporter = RichProgressReporter(progress, description="plotting...")
        plot(
            df=df,
            save_path=plots_path,
            fig_cols=("drug", "roi"),
            x_col="event",
            y_col="value",
            hue_col="genotype",
            min_sample_n=config["stat-test"]["min_sample_n"],
            progress_reporter=reporter,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f'config file not found: "{args.config}"')
    with args.config.open("rb") as f:
        config = tomllib.load(f)

    pipeline(config)


if __name__ == "__main__":
    main()

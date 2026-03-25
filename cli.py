from pathlib import Path
import tomllib
import argparse
import polars as pl
from rich.logging import RichHandler
from rich.progress import Progress
import logging

from fuser import find_roi_ids, get_annotation, plot

from dataset import Dataset
from process import correlation
from progress_rich import RichProgressReporter
from schema import check_pk

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


def pipeline(config: dict):
    paths = config["paths"]
    parameters = config["parameters"]

    fus_region_values_path = Path(paths["fus_region_values"])
    if not fus_region_values_path.is_file():
        with open(paths["roi"], "r") as f:
            rois = [l for l in f.read().splitlines() if l]

        roi_ids = find_roi_ids(
            rois=rois,
            ontology_path=paths["ontology"],
        )

        annotation_data, annotation_transform = get_annotation(
            annotation_path=paths["annotation"]
        )

        dataset = Dataset(
            root_path=paths["dataset"],
            session_path=paths["session"],
            event_path=paths["event"],
        )
        with Progress() as progress:
            reporter = RichProgressReporter(
                progress, description="processing raw fUS scans..."
            )
            df = correlation(
                dataset=dataset,
                roi_ids=roi_ids,
                annotation_data=annotation_data,
                annotation_transform=annotation_transform,
                voxel_percentile_thresh=parameters["voxel_percentile_thresh"],
                valid_region_voxel_ratio=parameters["valid_region_voxel_ratio"],
                hemodynamic_lag=parameters["hemodynamic_lag"],
                max_event_n=parameters["max_event_n"],
                min_event_time=parameters["min_event_time"],
                max_event_time=parameters["max_event_time"],
                post_event_exclusion_window=parameters["post_event_exclusion_window"],
                event_name="social",
                non_event_name="non-social",
                progress_reporter=reporter,
            )
        df.write_parquet(fus_region_values_path)
        logger.info(f'processed fus data saved to "{fus_region_values_path}"')
    else:
        logger.info(f'loading processed fus data from "{fus_region_values_path}"')
        df = pl.read_parquet(fus_region_values_path)

    check_pk(df, ("session", "event", "roi"))

    # fetch additional information for sessions
    session = pl.read_csv(paths["session"])
    check_pk(session, "fus")
    df = df.join(session, left_on="session", right_on="fus", how="left")

    genotype = pl.read_csv(paths["genotype"])
    check_pk(genotype, "subject")
    df = df.join(genotype, on="subject", how="left")

    plots_path = Path(paths["plots"])
    with Progress() as progress:
        reporter = RichProgressReporter(progress, description="plotting...")
        plot(
            df=df,
            save_path=plots_path,
            fig_cols=("drug", "roi"),
            x_col="event",
            y_col="value",
            hue_col="genotype",
            min_sample_n=parameters["min_sample_n"],
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

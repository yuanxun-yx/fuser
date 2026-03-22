from pathlib import Path
import tomllib
import argparse
import polars as pl
from rich.logging import RichHandler
from rich.progress import Progress
import logging

from dataset import Dataset
from process import process_fus
from ontology import find_roi_ids
from annotation import get_annotation
from analyze import plot
from progress_rich import RichProgressReporter

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


def pipeline(config: dict):
    paths = config['paths']
    parameters = config['parameters']

    fus_region_values_path = Path(paths['fus_region_values'])
    if not fus_region_values_path.is_file():
        roi_ids = find_roi_ids(
            ontology_path=paths['ontology'],
            roi_path=paths['roi'],
        )

        annotation_data, annotation_transform = get_annotation(annotation_path=paths['annotation'])

        dataset = Dataset(paths['dataset'])
        with Progress() as progress:
            reporter = RichProgressReporter(progress, description='processing raw fUS scans...')
            df = process_fus(
                dataset=dataset,
                roi_ids=roi_ids,
                annotation_data=annotation_data,
                annotation_transform=annotation_transform,
                voxel_percentile_thresh=parameters['voxel_percentile_thresh'],
                valid_region_voxel_ratio=parameters['valid_region_voxel_ratio'],
                hemodynamic_lag=parameters['hemodynamic_lag'],
                max_event_n=parameters['max_event_n'],
                min_event_time=parameters['min_event_time'],
                max_event_time=parameters['max_event_time'],
                post_event_exclusion_window=parameters['post_event_exclusion_window'],
                pca_n_components=parameters['pca_n_components'],
                progress_reporter=reporter,
            )
        df.write_parquet(fus_region_values_path)
        logger.info(f'processed fus data saved to "{fus_region_values_path}"')
    else:
        logger.info(f"loading processed fus data from {fus_region_values_path}")
        df = pl.read_parquet(fus_region_values_path)

    genotype = pl.read_csv(paths['genotype'])
    df = df.join(genotype, on='subject', how='left')

    plots_path = Path(paths['plots'])
    with Progress() as progress:
        reporter = RichProgressReporter(progress, description='plotting...')
        plot(
            df=df,
            save_path=plots_path,
            fig_cols=('drug', 'roi'),
            x_col='epoch_condition',
            y_col='value',
            hue_col='genotype',
            min_sample_n=parameters['min_sample_n'],
            progress_reporter=reporter,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.toml'))
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f'config file not found: {args.config}')
    with args.config.open('rb') as f:
        config = tomllib.load(f)

    pipeline(config)


if __name__ == "__main__":
    main()

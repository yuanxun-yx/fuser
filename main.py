from pathlib import Path
import tomllib
import argparse
import json
from typing import Any

from download import download_annotation_volume, download_allen_ontology
from process import process_fus
from analyze import plot


def pipeline(config: dict):
    paths = config['paths']
    parameters = config['parameters']

    annotation_path = Path(paths['annotation'])
    if not annotation_path.is_file():
        download_annotation_volume(annotation_path)

    fus_region_values_path = Path(paths['fus_region_values'])
    if not fus_region_values_path.is_file():
        print("processing fus raw data")
        process_fus(
            dataset_path=paths['dataset'],
            annotation_path=annotation_path,
            save_path=fus_region_values_path,
            voxel_percentile_thresh=parameters['voxel_percentile_thresh'],
            voxel_norm_percentile=parameters['voxel_norm_percentile'],
            valid_region_voxel_ratio=parameters['valid_region_voxel_ratio'],
            valid_region_pose_ratio=parameters['valid_region_pose_ratio'],
            fus_delay_s=parameters['fus_delay_s'],
        )
        print(f'processed fus data saved to "{fus_region_values_path}"')

    ontology_path = Path(paths['ontology'])
    if not ontology_path.is_file():
        download_allen_ontology(ontology_path, 1)  # adult mouse

    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    regions = {}
    queue = [ontology]
    while queue:
        item = queue.pop(0)
        regions[item['id']] = item  # reference in tree
        queue += item['children']

    def get_title(group: tuple[Any, ...]) -> str:
        drug, r_id = group
        region = regions[r_id]['acronym'].replace('/', '')
        return f'{region}+{drug}'

    plots_path = Path(paths['plots'])
    print(f'plotting, saving to "{plots_path}"')
    plot(
        data_path=fus_region_values_path,
        save_path=plots_path,
        fig_cols=('drug', 'brain_region_id'),
        x_col='epoch_condition',
        y_col='value',
        hue_col='genotype',
        min_sample_n=parameters['min_sample_n'],
        group_to_title=get_title,
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

from pathlib import Path
import tomllib
import argparse
import json
from typing import Any
import polars as pl
import nrrd
import numpy as np

from dataset import Dataset
from download import download_annotation_volume, download_allen_ontology
from process import process_fus
from analyze import plot


def pipeline(config: dict):
    paths = config['paths']
    parameters = config['parameters']

    ontology_path = Path(paths['ontology'])
    if not ontology_path.is_file():
        print(f'downloading ontology to "{ontology_path}"')
        download_allen_ontology(ontology_path, 1)  # adult mouse

    print(f"building region dict")
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    regions = {}
    queue = [ontology]
    while queue:
        item = queue.pop(0)
        regions[item['id']] = item  # reference in tree
        queue += item['children']

    fus_region_values_path = Path(paths['fus_region_values'])
    if not fus_region_values_path.is_file():
        annotation_path = Path(paths['annotation'])
        if not annotation_path.is_file():
            print(f'downloading annotation to "{annotation_path}"')
            download_annotation_volume(annotation_path)

        print(f'loading annotation from "{annotation_path}"')
        data, header = nrrd.read(str(annotation_path))
        print(f'getting ids from "{annotation_path}"')
        ids = np.unique(data)

        level = parameters['st_level']
        print(f"converting regions to highest parents below st level {level}")
        lut = np.arange(ids.max() + 1)
        for r in ids:
            if r == 0:
                continue
            d = regions[r]
            while d['st_level'] > level:
                p = regions[d['parent_structure_id']]
                # use d directly to keep regions disjoint
                # because if we use the region higher than level (p),
                # it might have children that's not mapped to this level (p's level)
                if p['st_level'] < level:
                    break
                d = p
            lut[r] = d['id']
        data = lut[data]

        print("processing fus raw data")
        dataset = Dataset(paths['dataset'])
        df = process_fus(
            dataset=dataset,
            annotation_header=header,
            annotation_data=data,
            voxel_percentile_thresh=parameters['voxel_percentile_thresh'],
            valid_region_voxel_ratio=parameters['valid_region_voxel_ratio'],
            fus_delay_s=parameters['fus_delay_s'],
        )
        print(f'processed fus data saved to "{fus_region_values_path}"')
    else:
        print(f"loading processed fus data from {fus_region_values_path}")
        df = pl.read_parquet(fus_region_values_path)

    def get_title(group: tuple[Any, ...]) -> str:
        drug, r_id = group
        region = regions[r_id]['acronym'].replace('/', '')
        return f'{region}+{drug}'

    genotype = pl.read_csv(paths['genotype'])
    df = df.join(genotype, on='subject', how='left')

    plots_path = Path(paths['plots'])
    print(f'plotting, saving to "{plots_path}"')
    plot(
        df=df,
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

from pathlib import Path
import tomllib
import argparse

from download import download_annotation_volume
from process import process_fus


def pipeline(config: dict):
    annotation_path = Path(config['paths']['annotation'])
    if not annotation_path.is_file():
        download_annotation_volume(annotation_path)

    fus_region_values_path = Path(config['paths']['fus_region_values'])
    if not fus_region_values_path.is_file():
        process_fus(
            dataset_path=config['paths']['dataset'],
            annotation_path=annotation_path,
            save_path=fus_region_values_path,
            **config['parameters']
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

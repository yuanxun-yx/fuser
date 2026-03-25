from pathlib import Path
import nrrd
import numpy as np
import logging

from .paths import get_cache_dir
from .download import download_annotation_volume
from .affine import check_valid_transform

logger = logging.getLogger(__name__)

FILENAME = "annotation.nrrd"


def load_annotation(
    ccf_version: int = 2022, resolution: int = 10, path: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path) if path else get_cache_dir() / FILENAME

    if not path.is_file():
        logger.info(f'downloading annotation to "{path}"')
        download_annotation_volume(path, ccf_version=ccf_version, resolution=resolution)

    logger.info(f'loading annotation from "{path}"')
    annotation_data, annotation_header = nrrd.read(str(path))

    if not np.all(annotation_header["sizes"] == annotation_data.shape):
        raise ValueError(
            f'"{path}" header shape {annotation_header["sizes"]} '
            f"does not match data shape {annotation_data.shape}"
        )

    annotation_transform = np.empty((4, 4))
    annotation_transform[3, :] = (0, 0, 0, 1)
    annotation_transform[:3, :3] = annotation_header["space directions"]
    annotation_transform[:3, 3] = annotation_header["space origin"]
    check_valid_transform(annotation_transform)

    return annotation_data, annotation_transform

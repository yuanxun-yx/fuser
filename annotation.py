from pathlib import Path
import nrrd
import numpy as np
import logging

from download import download_annotation_volume
from affine import check_valid_transform

logger = logging.getLogger(__name__)


def get_annotation(annotation_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        logger.info(f'downloading annotation to "{annotation_path}"')
        download_annotation_volume(annotation_path)

    logger.info(f'loading annotation from "{annotation_path}"')
    annotation_data, annotation_header = nrrd.read(str(annotation_path))

    if not np.all(annotation_header["sizes"] == annotation_data.shape):
        raise ValueError(
            f'"{annotation_path}" header shape {annotation_header["sizes"]} '
            f"does not match data shape {annotation_data.shape}"
        )

    annotation_transform = np.empty((4, 4))
    annotation_transform[3, :] = (0, 0, 0, 1)
    annotation_transform[:3, :3] = annotation_header["space directions"]
    annotation_transform[:3, 3] = annotation_header["space origin"]
    check_valid_transform(annotation_transform)

    return annotation_data, annotation_transform

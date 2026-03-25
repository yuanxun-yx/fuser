import numpy as np
from numpy.linalg import inv
from scipy.ndimage import affine_transform

# in um
BRAIN_ORIGIN_CCFv3_COORD_POST_INF_RIGHT = (5500, 5300, 6100)

# Iconeous "brain" coordinate
# direction: left, anterior, superior (dorsal)
# origin: approx 4mm below Bregma
# CCFv3 axis order: posterior, inferior, right
# 'space' in header is incorrect
# source: https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas
BRAIN_TO_ANNOTATION = np.zeros((4, 4))
BRAIN_TO_ANNOTATION[3, 3] = 1
BRAIN_TO_ANNOTATION[0, 1] = -1
BRAIN_TO_ANNOTATION[1, 2] = -1
BRAIN_TO_ANNOTATION[2, 0] = -1
BRAIN_TO_ANNOTATION[:3, 3] = BRAIN_ORIGIN_CCFv3_COORD_POST_INF_RIGHT
# magic number from Iconeous: 4mm (4000 um)
BRAIN_TO_ANNOTATION[:3, :3] *= 4e3


def transform(
    annotation_data: np.ndarray,
    shape: tuple[int, int, int, int],
    *,
    annotation_transform: np.ndarray,
    brain_to_lab: np.ndarray,
    probe_to_lab: np.ndarray,
    voxels_to_probe: np.ndarray,
):
    """
    Although upscaling the fUS data to atlas is common, we apply downscale to atlas annotations.
    This avoids disturbance to statistical testing caused by interpolating fUS raw data. Downscaling
    is more reliable than upscaling. Consider using soft masks when dealing with small brain regions.
    """
    if len(shape) != 4:
        raise ValueError("length of shape must be 4")

    voxels_to_annotation_index = (
        inv(annotation_transform)
        @ BRAIN_TO_ANNOTATION
        @ inv(brain_to_lab)
        @ probe_to_lab
        @ voxels_to_probe
    )

    # (pose, x, y, z) (no scan because only depend on pose)
    voxel_annotations = np.empty(shape, dtype=annotation_data.dtype)
    for i in range(shape[0]):
        voxel_annotations[i] = affine_transform(
            annotation_data,
            matrix=voxels_to_annotation_index[i],
            output_shape=shape[1:],
            order=0,  # nearest neighbor
        )

    return voxel_annotations

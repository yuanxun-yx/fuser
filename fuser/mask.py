import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_closing


def compute_valid_mask(
    data: np.ndarray, *, thresh: float, ignore_axis="y"
) -> np.ndarray:
    # ignore scan direction (y) because it doesn't include full size
    axes = (0, "xyz".index(ignore_axis) + 1)
    mean = data.mean(axis=0)
    threshold = np.percentile(mean, thresh)
    # (pose, x, y, z)
    mask = mean > threshold
    # morphology process each 3d mask for each pose
    for i in range(mask.shape[0]):
        labels, num = label(mask[i, ...])
        sizes = np.bincount(labels.ravel())
        largest_label = np.argmax(sizes[1:]) + 1
        body_mask = labels == largest_label
        mask[i, ...] = binary_closing(
            binary_fill_holes(body_mask, axes=axes), axes=axes
        )
    return mask

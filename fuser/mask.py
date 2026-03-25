import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_closing

# ignore scan direction (y) because it usually doesn't include full size
APPLY_AXES = (0, 2)


def compute_valid_mask(data: np.ndarray, *, thresh: float) -> np.ndarray:
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
            binary_fill_holes(body_mask, axes=APPLY_AXES), axes=APPLY_AXES
        )
    return mask

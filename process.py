from pathlib import Path
from typing import Any
import polars as pl
import numpy as np
from numpy.linalg import inv
from scipy.ndimage import affine_transform, label, binary_fill_holes, binary_closing
import warnings
import json
import nrrd

from dataset import Dataset
from download import download_allen_ontology, download_annotation_volume
from utils import check_valid_transform

type RoiIds = dict[str, list[int]]

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


def bincount_axes(
        x: np.ndarray,
        /,
        axis: int | tuple[int, ...] | None = None,
        weights: np.ndarray = None,
) -> np.ndarray:
    """
    apply numpy.bincount on given axes of x
    """
    if weights is not None:
        weights = np.broadcast_to(weights, x.shape)

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    # move flattened axes to the end
    dest = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, dest)

    batch_shape = x.shape[:-len(axis)]
    reduce_size = np.prod(x.shape[-len(axis):])

    x = x.reshape(-1, reduce_size)

    k = x.max() + 1
    b = x.shape[0]

    offset = np.arange(b)[:, None] * k
    x = (x + offset).ravel()

    # apply same flatten to weights
    if weights is not None:
        weights = np.moveaxis(weights, axis, dest)
        weights = weights.reshape(-1, reduce_size)
        weights = weights.ravel()

    counts = np.bincount(x, weights=weights, minlength=b * k)

    counts = counts.reshape(*batch_shape, k)

    return counts


def process_fus(
        dataset: Dataset,
        *,
        roi_ids: RoiIds,
        annotation_path: str | Path,
        voxel_percentile_thresh: float,
        valid_region_voxel_ratio: float,
        hemodynamic_lag: float,
) -> pl.DataFrame:
    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        print(f'downloading annotation to "{annotation_path}"')
        download_annotation_volume(annotation_path)

    print(f'loading annotation from "{annotation_path}"')
    annotation_data, annotation_header = nrrd.read(str(annotation_path))

    annotation_transform = np.empty((4, 4))
    annotation_transform[3, :] = (0, 0, 0, 1)
    annotation_transform[:3, :3] = annotation_header['space directions']
    annotation_transform[:3, 3] = annotation_header['space origin']
    check_valid_transform(annotation_transform)

    dfs = []
    # vectorize this in the future, size of session is ~200MB
    for session in dataset:
        fus_scan = session.fus_scan

        n_block_repeat = fus_scan.data.shape[2]
        if n_block_repeat != 1:
            raise NotImplementedError(f'block repeat number is {n_block_repeat}, we only handle 1 currently')
        data = fus_scan.data.squeeze(2)

        time = fus_scan.acquisition.time.squeeze(2)
        time += hemodynamic_lag

        # each pose + 3d correspond to one voxel in fixed space, not 3d alone
        # because probe is moving, think as 4d coordinate
        # check if each 4d point is valid
        data_avg_scans = data.mean(axis=0)
        threshold = np.percentile(data_avg_scans, voxel_percentile_thresh)
        # shape: pose, x, y, z
        mask = data_avg_scans > threshold
        # morphology process each 3d mask
        for i in range(mask.shape[0]):
            s = mask[i]
            labels, num = label(s)
            sizes = np.bincount(labels.ravel())
            largest_label = np.argmax(sizes[1:]) + 1
            body_mask = labels == largest_label
            # ignore scan direction because it has much smaller size
            for k in range(body_mask.shape[1]):
                s[:, k, :] = binary_closing(binary_fill_holes(body_mask[:, k, :]))

        r = session.processed
        r = np.swapaxes(r, 2, 1)
        r = r[..., None, :]

        voxels_to_annotation_index = (
                inv(annotation_transform) @ BRAIN_TO_ANNOTATION @ inv(session.brain_to_lab) @
                fus_scan.acquisition.probe_to_lab @ fus_scan.acquisition.voxels_to_probe
        )

        # shape: pose, x, y, z (no scan repeat because only depend on pose)
        voxel_annotations = np.empty(data.shape[1:], dtype=annotation_data.dtype)
        for i in range(data.shape[1]):
            voxel_annotations[i] = affine_transform(
                annotation_data,
                matrix=voxels_to_annotation_index[i],
                output_shape=data.shape[-3:],
                order=0  # nearest neighbor
            )

        # convert non-consecutive region ids to 0, 1, 2, ...
        ids, inverse = np.unique(voxel_annotations, return_inverse=True)
        # shape: id count
        region_voxel_count = bincount_axes(inverse)
        region_valid_voxel_count = bincount_axes(inverse, weights=mask)
        # shape: event, id count
        inverse_b = np.broadcast_to(inverse, r.shape)
        region_valid_value_sum = bincount_axes(inverse_b, axis=tuple(range(-4, 0)), weights=mask[None, ...] * r)

        for roi, subtree in roi_ids.items():
            m = np.isin(ids, subtree)
            roi_count = region_voxel_count[m].sum()
            if roi_count == 0:
                continue
            valid_ratio = region_valid_voxel_count[m].sum() / roi_count
            if valid_ratio < valid_region_voxel_ratio:
                continue
            valid_value_mean = region_valid_value_sum[:, m].sum(axis=1) / region_valid_voxel_count[m].sum()
            for i, k in enumerate(['social', 'non-social']):
                dfs.append({
                    'session': fus_scan.metadata.file_id,
                    'subject': session.subject,
                    **{name: cond for name, cond in zip(dataset.CONDITION_NAMES, session.conditions)},
                    'epoch_condition': k,
                    'roi': roi,
                    'value': valid_value_mean[i],
                })

    df = pl.DataFrame(dfs)
    return df


def find_subtree(root: dict[str, Any]) -> RoiIds:
    subtree = {}

    def dfs(node):
        nodes = [node['id']]
        for c in node['children']:
            nodes.extend(dfs(c))
        subtree[node['acronym']] = nodes
        return nodes

    dfs(root)
    return subtree


def find_roi_ids(
        ontology_path: str | Path,
        roi_path: str | Path,
) -> RoiIds:
    ontology_path = Path(ontology_path)
    roi_path = Path(roi_path)

    with open(roi_path, 'r') as f:
        rois = set(l for l in f.read().splitlines() if l)

    if not ontology_path.is_file():
        download_allen_ontology(ontology_path, 1)  # adult mouse

    with open(ontology_path, 'r') as f:
        ontology = json.load(f)

    subtree = find_subtree(ontology)

    valid_rois = []
    invalid_rois = []

    for r in rois:
        if r in subtree:
            valid_rois.append(r)
        else:
            invalid_rois.append(r)

    if invalid_rois:
        warnings.warn(f"{invalid_rois} are not found in structure tree")

    roi_ids = {k: subtree[k] for k in valid_rois}

    return roi_ids

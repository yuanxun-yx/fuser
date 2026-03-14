from pathlib import Path
import tomllib
import argparse
from tqdm import tqdm
import json
import logging
from typing import Literal
import requests
import numpy as np
from numpy.linalg import inv
import openpyxl
from iconeus_scan import read_scan, read_bps
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import nrrd
from scipy.ndimage import affine_transform

# in um
BRAIN_ORIGIN_CCFv3_COORD_POST_INF_ML = (5600, 5500, 5700)


def find_only_file(root: Path, prefix: str, postfix: str) -> Path:
    pattern = f'{prefix}*{postfix}'
    result = list(root.glob(pattern))

    if len(result) == 0:
        raise ValueError(f'"{root}" does not contain {pattern}')
    if len(result) > 1:
        logging.warning(f'"{root}" contains {len(result)} of {pattern}, first one used')

    return result[0]


def download_annotation_volume(
        file_name: Path,
        ccf_version: Literal[2015, 2016, 2017, 2022] = 2022,
        resolution: Literal[10, 25, 50, 100] = 10
):
    url = (f'https://download.alleninstitute.org/informatics-archive/current-release/'
           f'mouse_ccf/annotation/ccf_{ccf_version}/annotation_{resolution}.nrrd')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_name, "wb") as f:
            for chunk in r.iter_content():
                f.write(chunk)


def download_allen_ontology(
        file_name: Path,
        structure_id: int
):
    url = f'https://api.brain-map.org/api/v2/structure_graph_download/{structure_id}.json'
    with requests.get(url) as r:
        r.raise_for_status()
        content = json.loads(r.content)
        if not content['success']:
            raise ValueError(f'ontology download failed')
        with open(file_name, "w") as f:
            json.dump(content['msg'][0], f)


def read_event_time(xlsx_path: Path):
    SOCIAL_EVENT_NAME = 'Modified Social Event (Horizontal)'
    NONSOCIAL_EVENT_NAME = 'Modified Non-Social Event (Horizontal)'

    wb = openpyxl.load_workbook(xlsx_path)
    if len(wb.sheetnames) == 1:
        ws = wb.worksheets[0]
    else:
        ws = wb['Processed Events']
    if ws.max_column != 2:
        raise ValueError(f'"{xlsx_path}" has {ws.max_column} columns')
    social_events, nonsocial_events = None, None
    for i in range(2, ws.max_row + 1):
        event_type = ws.cell(row=i, column=1).value
        if event_type not in [SOCIAL_EVENT_NAME, NONSOCIAL_EVENT_NAME]: continue
        values = ws.cell(row=i, column=2).value
        times = [int(s) for s in values.split(' ') if s]
        if len(times) % 2 != 0:
            logging.warning(f'number of timepoints {len(times)} is not even')
        events = list(zip(times[::2], times[1::2]))
        if event_type == SOCIAL_EVENT_NAME:
            if social_events is not None:
                raise ValueError(f'more than one social event found in "{xlsx_path}"')
            social_events = events
        if event_type == NONSOCIAL_EVENT_NAME:
            if nonsocial_events is not None:
                raise ValueError(f'more than one non-social event found in "{xlsx_path}"')
            nonsocial_events = events
    return social_events, nonsocial_events


def pipeline(config: dict):
    ontology_path = Path(config['paths']['ontology'])
    if not ontology_path.is_file():
        download_allen_ontology(ontology_path, 1)  # adult mouse
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    regions = {}
    queue = [ontology]
    while len(queue) > 0:
        item = queue.pop(0)
        regions[item['id']] = item  # reference in tree
        queue += item['children']

    annotation_path = Path(config['paths']['annotation'])
    if not annotation_path.is_file():
        download_annotation_volume(annotation_path)
    annotation_data, annotation_header = nrrd.read(annotation_path)

    annotation_transform = np.eye(4)
    annotation_transform[:3, :3] = annotation_header['space directions']
    annotation_transform[:3, 3] = annotation_header['space origin']

    # Iconeous "brain" coordinate is stereotaxic
    # direction: left, anterior, superior (dorsal)
    # origin: approx 4mm below Bregma
    # CCFv3 axis order: posterior, inferior, right
    # 'space' in header is incorrect
    # source: https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas
    brain_to_annotation = np.zeros((4, 4))
    brain_to_annotation[3, 3] = 1
    brain_to_annotation[0, 1] = -1
    brain_to_annotation[1, 2] = -1
    brain_to_annotation[2, 0] = -1
    brain_to_annotation[:3, 3] = BRAIN_ORIGIN_CCFv3_COORD_POST_INF_ML
    # magic number from Iconeous: 4mm (4000 um)
    brain_to_annotation[:3, :3] *= 4e3

    data_root = Path(config['paths']['data_root']).expanduser().resolve()
    for timepoint_dir in data_root.iterdir():
        if not timepoint_dir.is_dir(): continue
        timepoint = timepoint_dir.name
        event_time_dir = timepoint_dir / 'eventTime'
        for group_dir in event_time_dir.iterdir():
            if not group_dir.is_dir(): continue
            scan_dir = timepoint_dir / 'Scan' / group_dir.name.replace('eventTime_', 'Scan_')
            if not scan_dir.is_dir():
                raise ValueError(f'scan dir does not exist: "{scan_dir}"')
            group = group_dir.name.split('_')[2]
            # tqdm(list(group_dir.glob('*.xlsx')), desc=f"{timepoint},{group}")
            for event_time_path in group_dir.glob('[!~]*.xlsx'):
                # each trial
                parts = event_time_path.stem.split('_')
                subject = parts[0]
                prefix = '_'.join(parts[:-3])

                event_times = read_event_time(event_time_path)

                bps_path = find_only_file(scan_dir, prefix, '.source.bps')
                brain_to_lab = read_bps(bps_path)

                fus_scan_path = find_only_file(scan_dir, prefix, 'fus3D.source.scan')
                fus_scan = read_scan(fus_scan_path)

                t = 0
                frame = fus_scan.data[:, :, :, t]


                voxels_to_annotation_index = (
                        inv(annotation_transform) @ brain_to_annotation @ inv(brain_to_lab) @
                        fus_scan.probe_to_lab[t, :, :] @ fus_scan.voxels_to_probe
                )

                # threshold = 10.
                # coords = np.argwhere(frame > threshold)
                # voxels_to_brain = inv(brain_to_lab) @ fus_scan.probe_to_lab[t, :, :] @ fus_scan.voxels_to_probe
                # indices = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
                # out = (voxels_to_brain @ indices.T).T
                # world = out[:, :3]
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(world[:, 0], world[:, 1], world[:, 2], c=frame[tuple(coords.T)], s=1)
                # ax.set_xlabel('x')
                # ax.set_ylabel('y')
                # ax.set_zlabel('z')
                # ax.set_aspect('equal')
                # plt.show()

                voxel_annotations = affine_transform(
                    annotation_data,
                    matrix=voxels_to_annotation_index,
                    output_shape=frame.shape,
                    order=0  # nearest neighbor
                )

                n_slice = frame.shape[1]
                for i in range(n_slice):
                    s = voxel_annotations[:,i,:].T
                    plt.contour(s, levels=np.unique(s)[1:], colors='red', linewidths=.1)
                    plt.imshow(frame[:,i,:].T)
                    plt.savefig(Path('align') / f'{prefix}_{i}.png')
                    plt.clf()


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

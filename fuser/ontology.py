from pathlib import Path
from typing import Any
import logging
import json

from .download import download_allen_ontology

type RoiIds = dict[str, list[int]]

logger = logging.getLogger(__name__)


def find_subtree(root: dict[str, Any]) -> RoiIds:
    subtree = {}

    def dfs(node):
        nodes = [node["id"]]
        for c in node["children"]:
            nodes.extend(dfs(c))
        subtree[node["acronym"]] = nodes
        return nodes

    dfs(root)
    return subtree


def find_roi_ids(
    ontology_path: str | Path,
    roi_path: str | Path,
) -> RoiIds:
    ontology_path = Path(ontology_path)
    roi_path = Path(roi_path)

    with open(roi_path, "r") as f:
        rois = set(l for l in f.read().splitlines() if l)

    if not ontology_path.is_file():
        download_allen_ontology(ontology_path, 1)  # adult mouse

    with open(ontology_path, "r") as f:
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
        logging.warning(f"{invalid_rois} are not found in structure tree")

    roi_ids = {k: subtree[k] for k in valid_rois}

    return roi_ids

from pathlib import Path
from typing import Any
import warnings
import json
import logging

from .download import download_allen_ontology
from .paths import get_cache_dir

type RoiIds = dict[str, list[int]]

logger = logging.getLogger(__name__)

FILENAME = "ontology.json"


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
    rois: list[str],
    *,
    path: str | Path | None = None,
) -> RoiIds:
    path = Path(path) if path else get_cache_dir() / FILENAME

    if not path.is_file():
        logger.info(f'downloading ontology to "{path}"')
        download_allen_ontology(path, 1)  # adult mouse

    logger.info(f'loading ontology from "{path}"')
    with open(path, "r") as f:
        ontology = json.load(f)

    subtree = find_subtree(ontology)

    rois = set(rois)

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

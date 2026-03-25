import requests
import json
from pathlib import Path


def download_annotation_volume(
    file_name: str | Path,
    *,
    ccf_version: int,
    resolution: int,
) -> None:
    file_name = Path(file_name)
    url = (
        f"https://download.alleninstitute.org/informatics-archive/current-release/"
        f"mouse_ccf/annotation/ccf_{ccf_version}/annotation_{resolution}.nrrd"
    )
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        file_name.write_bytes(r.content)


def download_allen_ontology(file_name: str | Path, structure_id: int) -> None:
    file_name = Path(file_name)
    url = (
        f"https://api.brain-map.org/api/v2/structure_graph_download/{structure_id}.json"
    )
    with requests.get(url) as r:
        r.raise_for_status()
        content = json.loads(r.content)
        if not content["success"]:
            raise RuntimeError("ontology download failed")
        with open(file_name, "w") as f:
            json.dump(content["msg"][0], f)

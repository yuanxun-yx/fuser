from pathlib import Path
from typing import Iterator
import logging
from dataclasses import dataclass
import numpy as np
import polars as pl
from bisect import bisect_left

from scan import read_scan, read_bps, Scan
from event import read_events

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Session:
    id: str
    subject: str
    conditions: tuple[str, ...]
    fus_scan: Scan
    brain_to_lab: np.ndarray
    events: np.ndarray


def match_prefix(prefix: str, files: list[str]) -> str | None:
    i = bisect_left(files, prefix)
    if i < len(files) and files[i].startswith(prefix):
        return files[i]
    return None


class Dataset:
    def __init__(
        self,
        root_path: str | Path,
        session_path: str | Path,
        event_path: str | Path,
    ) -> None:
        self._root_path = Path(root_path)

        scan_files = sorted(p.name for p in self._root_path.glob("*.source.scan"))
        bps_files = sorted(p.name for p in self._root_path.glob("*.source.bps"))

        df = pl.read_csv(session_path)
        self.condition_names = tuple(
            c for c in df.columns if c not in ("fus", "bps", "subject")
        )

        df = df.with_columns(
            [
                pl.col("fus")
                .map_elements(lambda s: match_prefix(s, scan_files))
                .alias("fus_file"),
                pl.col("bps")
                .map_elements(lambda s: match_prefix(s, bps_files))
                .alias("bps_file"),
            ]
        )

        df_null = df.filter(
            pl.any_horizontal(
                [
                    pl.col("fus_file").is_null(),
                    pl.col("bps_file").is_null(),
                ]
            )
        )
        for r in df_null.iter_rows(named=True):
            if r["fus_file"].is_null():
                logger.warning(f'fail to find scan file with prefix tag "{r["fus"]}"')
            if r["bps_file"].is_null():
                logger.warning(f'fail to find bps file with prefix tag "{r["bps"]}"')

        df = df.filter(
            pl.all_horizontal(
                [
                    pl.col("fus").is_not_null(),
                    pl.col("bps").is_not_null(),
                ]
            )
        )

        event = read_events(event_path)
        # temporarily use subject
        df = df.join(event, on="subject", how="left")
        df_null = df.filter(
            pl.col("times").is_null(),
        )
        for r in df_null.iter_rows(named=True):
            logger.warning(f'no event times for fus scan "{r["fus"]}"')
        self._df = df.filter(
            pl.col("times").is_not_null(),
        )

    def __iter__(self) -> Iterator[Session]:
        for r in self._df.iter_rows(named=True):
            fus_scan = read_scan(self._root_path / r["fus_file"])
            brain_to_lab = read_bps(self._root_path / r["bps_file"])
            conditions = tuple(r[k] for k in self.condition_names)
            events = np.array(r["times"]).reshape(-1, 2)
            yield Session(
                id=r["fus"],
                fus_scan=fus_scan,
                brain_to_lab=brain_to_lab,
                subject=r["subject"],
                conditions=conditions,
                events=events,
            )

    def __len__(self) -> int:
        return len(self._df)

from pathlib import Path
from typing import Iterator
import warnings
from dataclasses import dataclass
import numpy as np

from scan import read_scan, read_bps, Scan
from epochs import read_epochs


@dataclass(frozen=True)
class SessionMetadata:
    fus_scan_path: Path
    bps_path: Path
    epochs_path: Path
    subject: str
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class Session:
    subject: str
    conditions: tuple[str, ...]
    fus_scan: Scan
    brain_to_lab: np.ndarray
    epochs: np.ndarray


class Dataset:
    CONDITION_NAMES = ('drug',)

    def __init__(self, path: str | Path, show_progress: bool = True) -> None:
        self.path = Path(path)
        self.show_progress = show_progress
        self.sessions = []

        for drug_dir in self.path.iterdir():
            if not drug_dir.is_dir(): continue
            drug = drug_dir.name.lower()
            epochs_dir = drug_dir / 'eventTime'
            scan_dir = drug_dir / 'Scan'
            if not scan_dir.is_dir():
                warnings.warn(f'scan folder "{scan_dir}" does not exist, skipping')
                continue
            for epochs_path in epochs_dir.glob('[!~]*.xlsx'):
                parts = epochs_path.stem.split('_')
                subject = parts[0].lower()
                prefix = '_'.join(parts[:-3])
                try:
                    bps_path = self._find_only_file(scan_dir, f'{prefix}*.source.bps')
                    fus_scan_path = self._find_only_file(scan_dir, f'{prefix}*fus3D.source.scan')
                except FileNotFoundError as e:
                    warnings.warn(str(e))
                    continue
                self.sessions.append(
                    SessionMetadata(
                        fus_scan_path=fus_scan_path,
                        bps_path=bps_path,
                        epochs_path=epochs_path,
                        subject=subject,
                        conditions=(drug,),
                    )
                )

    def __iter__(self) -> Iterator[Session]:
        for s in self.sessions:
            fus_scan = read_scan(s.fus_scan_path)
            brain_to_lab = read_bps(s.bps_path)
            epochs = read_epochs(s.epochs_path)
            yield Session(
                fus_scan=fus_scan,
                brain_to_lab=brain_to_lab,
                subject=s.subject,
                epochs=epochs,
                conditions=s.conditions,
            )

    def __len__(self) -> int:
        return len(self.sessions)

    @staticmethod
    def _find_only_file(path: Path, pattern: str) -> Path:
        result = list(path.glob(pattern))

        if len(result) == 0:
            raise FileNotFoundError(f'"{path}" does not contain "{pattern}"')
        if len(result) > 1:
            warnings.warn(f'"{path}" contains {len(result)} of "{pattern}", first one used')

        return result[0]

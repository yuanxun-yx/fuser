from pathlib import Path
from typing import Iterator
from tqdm import tqdm
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
    epochs: dict[str, np.ndarray]


class Dataset:
    CONDITION_NAMES = ('drug', 'genotype')

    def __init__(self, path: str | Path, show_progress: bool = True):
        self.path = Path(path)
        self.show_progress = show_progress
        self.sessions = []

        for drug_dir in self.path.iterdir():
            if not drug_dir.is_dir(): continue
            drug = drug_dir.name
            epochs_dir = drug_dir / 'eventTime'
            for genotype_dir in epochs_dir.iterdir():
                if not genotype_dir.is_dir(): continue
                scan_dir = drug_dir / 'Scan' / genotype_dir.name.replace('eventTime_', 'Scan_')
                if not scan_dir.is_dir():
                    warnings.warn(f'scan folder "{scan_dir}" does not exist, skipping')
                    continue
                genotype = genotype_dir.name.split('_')[2]
                for epochs_path in genotype_dir.glob('[!~]*.xlsx'):
                    parts = epochs_path.stem.split('_')
                    subject = parts[0]
                    prefix = '_'.join(parts[:-3])
                    try:
                        bps_path = self._find_only_file(scan_dir, prefix, '.source.bps')
                        fus_scan_path = self._find_only_file(scan_dir, prefix, 'fus3D.source.scan')
                    except FileNotFoundError as e:
                        warnings.warn(str(e))
                        continue
                    self.sessions.append(
                        SessionMetadata(
                            fus_scan_path=fus_scan_path,
                            bps_path=bps_path,
                            epochs_path=epochs_path,
                            subject=subject,
                            conditions=(drug, genotype),
                        )
                    )

    def __iter__(self) -> Iterator[Session]:
        it = self.sessions
        if self.show_progress:
            it = tqdm(it)
        for s in it:
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

    @staticmethod
    def _find_only_file(path: Path, prefix: str, postfix: str) -> Path:
        pattern = f'{prefix}*{postfix}'
        result = list(path.glob(pattern))

        if len(result) == 0:
            raise FileNotFoundError(f'"{path}" does not contain "{pattern}"')
        if len(result) > 1:
            warnings.warn(f'"{path}" contains {len(result)} of "{pattern}", first one used')

        return result[0]

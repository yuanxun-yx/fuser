from typing import Protocol


class ProgressReporter(Protocol):
    def start(self, total: int) -> None: ...

    def advance(self, n: int = 1) -> None: ...

from rich.progress import Progress, TaskID


class RichProgressReporter:
    def __init__(self, progress: Progress, description: str | None = None) -> None:
        self._progress = progress
        self._description = description
        self._task_id: TaskID | None = None

    def start(self, total: int) -> None:
        kwargs = {"total": total, "completed": 0}
        if self._description is not None:
            kwargs["description"] = self._description
        self._task_id = self._progress.add_task(**kwargs)

    def advance(self, n: int = 1) -> None:
        if self._task_id is None:
            raise RuntimeError("start() must be called before update()")
        self._progress.update(self._task_id, advance=n)

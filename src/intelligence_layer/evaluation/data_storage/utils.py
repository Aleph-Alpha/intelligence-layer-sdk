from pathlib import Path


class FileBasedRepository:
    """An :class:`FileBasedRepository` that stores evaluation results in files.

    Args:
        root_directory: The folder where the files are stored. The folder
            (along with its parents) will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    @staticmethod
    def write_utf8(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def read_utf8(path: Path) -> str:
        return path.read_text(encoding="utf-8")

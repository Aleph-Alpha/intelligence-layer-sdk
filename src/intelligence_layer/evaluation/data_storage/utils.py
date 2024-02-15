from pathlib import Path


class FileBasedRepository:
    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    @staticmethod
    def write_utf8(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def read_utf8(path: Path) -> str:
        return path.read_text(encoding="utf-8")

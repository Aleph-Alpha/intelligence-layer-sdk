from abc import ABC
from pathlib import Path
from typing import cast

from fsspec import AbstractFileSystem  # type: ignore


class FileSystemBasedRepository(ABC):
    """An :class:`FileBasedRepository` that stores evaluation results in files.

    Args:
        root_directory: The folder where the files are stored. The folder
            (along with its parents) will be created if it does not exist yet.
    """

    def __init__(self, file_system: AbstractFileSystem, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory
        self._file_system = file_system

    def write_utf8(self, path: Path, content: str) -> None:
        self._file_system.write_text(self.path_to_str(path), content, encoding="utf-8")

    def read_utf8(self, path: Path) -> str:
        return cast(
            str, self._file_system.read_text(self.path_to_str(path), encoding="utf-8")
        )

    def exists(self, path: Path) -> bool:
        return cast(bool, self._file_system.exists(path))

    @staticmethod
    def path_to_str(path: Path) -> str:
        """Returns a string for the given Path so that it's readable for the respective file system.

        Args:
            path: Given Path that should be converted.
        Returns:
            String representation of the given Path.
        """
        return str(path)

from abc import ABC
from pathlib import Path
from typing import Sequence, cast

from fsspec import AbstractFileSystem  # type: ignore


class FileSystemBasedRepository(ABC):
    """An :class:`FileBasedRepository` that stores evaluation results in files.

    Args:
        root_directory: The folder where the files are stored. The folder
            (along with its parents) will be created if it does not exist yet.
    """

    def __init__(self, file_system: AbstractFileSystem, root_directory: Path) -> None:
        self._root_directory = root_directory
        self._file_system = file_system
        self.mkdir(root_directory)

    def write_utf8(
        self, path: Path, content: str, create_parents: bool = False
    ) -> None:
        if create_parents:
            self.mkdir(path.parent)
        self._file_system.write_text(self.path_to_str(path), content, encoding="utf-8")

    def read_utf8(self, path: Path) -> str:
        return cast(
            str, self._file_system.read_text(self.path_to_str(path), encoding="utf-8")
        )

    def exists(self, path: Path) -> bool:
        return cast(bool, self._file_system.exists(self.path_to_str(path)))

    def mkdir(self, path: Path) -> None:
        if self.exists(path):
            return
        self._file_system.makedir(self.path_to_str(path), create_parents=True)

    def file_names(self, path: Path, file_type: str = "json") -> Sequence[str]:
        # return [
        #     file.rsplit(".", 1)[0]
        #     for file in self._file_system.ls(self.path_to_str(path))
        #     if file.endswith("." + file_type)
        # ]
        files = [Path(file) for file in self._file_system.ls(self.path_to_str(path))]
        return [file.stem for file in files if file.suffix == "." + file_type]

    @staticmethod
    def path_to_str(path: Path) -> str:
        """Returns a string for the given Path so that it's readable for the respective file system.

        Args:
            path: Given Path that should be converted.
        Returns:
            String representation of the given Path.
        """
        return str(path)

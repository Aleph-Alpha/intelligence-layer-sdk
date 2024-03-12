from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from fsspec import AbstractFileSystem # type: ignore


class FileSystemBasedRepository(ABC):
    """An :class:`FileBasedRepository` that stores evaluation results in files.

    Args:
        root_directory: The folder where the files are stored. The folder
            (along with its parents) will be created if it does not exist yet.
    """
        

    def __init__(self, fs: AbstractFileSystem, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory
        self._fs = fs

    def write_utf8(self, path: Path, content: str) -> None:
        self._fs.write_text(self.path_to_str(path), content, encoding="utf-8")

    def read_utf8(self, path: Path) -> str:
        return cast(str, self._fs.read_text(self.path_to_str(path), encoding="utf-8"))
    
    
    @staticmethod
    @abstractmethod
    def path_to_str(path: Path) -> str:
        """Returns a string for the given Path so that it's readable for the respective file system.

        Args:
            path: Given Path that should be converted.
        Returns:
            String representation of the given Path.
        """
        ...

    


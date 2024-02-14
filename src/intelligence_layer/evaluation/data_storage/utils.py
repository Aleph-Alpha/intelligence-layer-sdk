from pathlib import Path

from intelligence_layer.core.tracer import FileTracer, InMemoryTracer


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_log(log_path: Path) -> InMemoryTracer:
    return FileTracer(log_path).trace()

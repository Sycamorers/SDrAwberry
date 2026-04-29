"""Small I/O helpers shared by the SDrAwberry command line tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence


def setup_logging(verbose: bool = False) -> None:
    """Configure a compact root logger for command line use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, description: str = "file") -> Path:
    """Return *path* if it exists as a file, otherwise raise a useful error."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_dir(path: Path, description: str = "directory") -> Path:
    """Return *path* if it exists as a directory, otherwise raise a useful error."""
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def parse_float_list(value: str | Sequence[float]) -> list[float]:
    """Parse a comma-separated float list or normalize an existing sequence."""
    if isinstance(value, str):
        try:
            parsed = [float(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Could not parse float list from {value!r}") from exc
    else:
        parsed = [float(x) for x in value]
    if not parsed:
        raise ValueError("Expected at least one numeric value.")
    return parsed


def write_lines(path: Path, lines: Iterable[str]) -> None:
    """Write newline-separated text to a UTF-8 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

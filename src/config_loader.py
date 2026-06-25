"""Minimal, self-contained config loader for evtol_frozen_baseline.

Finds ``config.yaml`` relative to this file, loads it, and resolves every path
under ``paths:`` to an absolute :class:`pathlib.Path` (relative paths are taken
relative to the folder root — the directory that contains config.yaml).

Optionally reads a ``.env`` at the folder root for ``DRIVE_PATH`` so absolute
data locations can live outside version control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Folder root = parent of this file's directory (src/ -> root).
FOLDER_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = FOLDER_ROOT / "config.yaml"


def _maybe_load_dotenv() -> None:
    """Load a .env at the folder root if python-dotenv is available."""
    env_path = FOLDER_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def _resolve_path(value: str) -> Path:
    """Resolve a configured path to an absolute Path.

    - Absolute paths are returned as-is.
    - ``$DRIVE_PATH`` / ``${DRIVE_PATH}`` and ``~`` are expanded.
    - Relative paths are taken relative to FOLDER_ROOT.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(expanded)
    if not p.is_absolute():
        p = FOLDER_ROOT / p
    return p.resolve()


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load config.yaml and resolve all paths under ``paths:`` to absolute Paths.

    Returns the parsed config dict. Adds a ``folder_root`` key (Path) for
    convenience. Leaves EDIT-ME placeholders untouched but still resolves them
    (so the resulting absolute path makes the missing value obvious).
    """
    _maybe_load_dotenv()

    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["folder_root"] = FOLDER_ROOT

    paths = cfg.get("paths", {})
    for key, value in list(paths.items()):
        if value is None:
            continue
        paths[key] = _resolve_path(value)
    cfg["paths"] = paths

    return cfg


if __name__ == "__main__":
    import pprint

    pprint.pprint(load_config())

#!/usr/bin/env python3
"""CLI wrapper for the SDrAwberry batching stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdrawberry.batch import build_arg_parser, config_from_args, prepare_and_run
from sdrawberry.config import load_yaml_config
from sdrawberry.io_utils import setup_logging


def _defaults_from_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg = load_yaml_config(Path(path))
    defaults = dict(cfg.get("batching", cfg) or {})
    if "scene_dir" not in defaults and "scene_dir" in cfg:
        defaults["scene_dir"] = cfg["scene_dir"]
    defaults.pop("enabled", None)
    return defaults


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, remaining = pre.parse_known_args()

    parser = build_arg_parser(defaults=_defaults_from_config(pre_args.config))
    args = parser.parse_args(remaining)
    setup_logging(verbose=bool(args.verbose))
    prepare_and_run(config_from_args(args))


if __name__ == "__main__":
    main()

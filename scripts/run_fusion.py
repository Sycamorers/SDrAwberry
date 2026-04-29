#!/usr/bin/env python3
"""CLI wrapper for the SDrAwberry fusion stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdrawberry.config import load_yaml_config
from sdrawberry.fusion import build_arg_parser, config_from_args, run_fusion
from sdrawberry.io_utils import setup_logging


def _defaults_from_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg = load_yaml_config(Path(path))
    defaults = dict(cfg.get("fusion", cfg) or {})
    if "base_dir" not in defaults and "scene_dir" in cfg:
        defaults["base_dir"] = cfg["scene_dir"]
    defaults.pop("enabled", None)
    return defaults


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, remaining = pre.parse_known_args()

    parser = build_arg_parser(defaults=_defaults_from_config(pre_args.config))
    args = parser.parse_args(remaining)
    setup_logging(verbose=bool(args.verbose))
    run_fusion(config_from_args(args))


if __name__ == "__main__":
    main()

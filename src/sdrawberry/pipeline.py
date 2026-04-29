"""Config-driven full SDrAwberry pipeline runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .batch import BatchConfig, parse_batch_range, prepare_and_run
from .clustering import ClusteringConfig, run_clustering
from .concave_hull import ConcaveHullConfig, run_concave_hull
from .config import load_yaml_config
from .fusion import FusionConfig, parse_range, run_fusion
from .io_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _stage_enabled(config: dict[str, Any], name: str, default: bool = True) -> bool:
    stages = config.get("stages", {})
    if isinstance(stages, dict) and name in stages:
        return bool(stages[name])
    section = config.get(name, {})
    if isinstance(section, dict) and "enabled" in section:
        return bool(section["enabled"])
    return default


def _path(value: Any, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("Missing required path value.")
        return default
    return Path(value)


def run_full_pipeline(config_path: Path) -> None:
    """Run configured pipeline stages from a YAML file."""
    cfg = load_yaml_config(Path(config_path))
    scene_dir = _path(cfg.get("scene_dir"))

    if _stage_enabled(cfg, "batching", default=True):
        options = dict(cfg.get("batching", {}) or {})
        options.pop("enabled", None)
        options["scene_dir"] = _path(options.get("scene_dir"), scene_dir)
        if isinstance(options.get("batch_range"), str):
            options["batch_range"] = parse_batch_range(options["batch_range"])
        LOGGER.info("Starting batching stage.")
        prepare_and_run(BatchConfig(**options))

    fusion_out = scene_dir / "merged" / "merged_global.ply"
    if _stage_enabled(cfg, "fusion", default=True):
        options = dict(cfg.get("fusion", {}) or {})
        options.pop("enabled", None)
        options["base_dir"] = _path(options.get("base_dir"), scene_dir)
        options["out"] = _path(options.get("out"), fusion_out)
        if isinstance(options.get("batch_range"), str):
            options["batch_range"] = parse_range(options["batch_range"])
        LOGGER.info("Starting fusion stage.")
        fusion_out = run_fusion(FusionConfig(**options))

    clustering_out = scene_dir / "clusters"
    if _stage_enabled(cfg, "clustering", default=True):
        options = dict(cfg.get("clustering", {}) or {})
        options.pop("enabled", None)
        options["input_pcd"] = _path(options.get("input_pcd"), fusion_out)
        options["out_dir"] = _path(options.get("out_dir"), clustering_out)
        LOGGER.info("Starting clustering stage.")
        run_clustering(ClusteringConfig(**options))
        clustering_out = Path(options["out_dir"])

    if _stage_enabled(cfg, "concave_hull", default=True):
        options = dict(cfg.get("concave_hull", {}) or {})
        options.pop("enabled", None)
        cluster_glob = str(options.pop("cluster_glob", "cluster_*.ply"))
        explicit_in = options.pop("in_pcd", None)
        base_mesh_dir = _path(options.pop("out_dir", None), scene_dir / "meshes")

        if explicit_in is not None:
            in_pcd = _path(explicit_in)
            out_dir = base_mesh_dir if base_mesh_dir.name != "meshes" else base_mesh_dir / in_pcd.stem
            LOGGER.info("Starting concave hull stage for %s.", in_pcd)
            run_concave_hull(ConcaveHullConfig(in_pcd=in_pcd, out_dir=out_dir, **options))
        else:
            cluster_files = sorted(Path(clustering_out).glob(cluster_glob))
            if not cluster_files:
                raise FileNotFoundError(f"No cluster point clouds matched {cluster_glob!r} in {clustering_out}")
            for cluster_pcd in cluster_files:
                out_dir = base_mesh_dir / cluster_pcd.stem
                LOGGER.info("Starting concave hull stage for %s.", cluster_pcd)
                run_concave_hull(ConcaveHullConfig(in_pcd=cluster_pcd, out_dir=out_dir, **options))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the full pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Run the SDrAwberry pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/example_scene.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``sdrawberry-full-pipeline``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(args.verbose))
    run_full_pipeline(Path(args.config))


if __name__ == "__main__":
    main()

"""Batch synchronized multi-camera image sequences and run DA3 per batch."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .colmap_io import export_points_ply_from_colmap_model
from .io_utils import ensure_dir, require_dir, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for image batching and optional DA3 execution."""

    scene_dir: Path
    batch_size: int = 10
    overlap_frames: int = 0
    symlink: bool = False
    only_batch: int | None = None
    max_batches: int | None = None
    cams_per_frame: int | None = None
    batch_range: tuple[int, int] | None = None
    prepare_only: bool = False
    da3_model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_device: str = "cuda"
    da3_conf_thresh_percentile: float = 75.0
    da3_num_max_points: int = 1_000_000
    da3_process_res: int = 504
    da3_process_res_method: str = "upper_bound_resize"
    da3_ref_view_strategy: str = "saddle_balanced"
    da3_export_ply: bool = False
    da3_disable_triton: bool = False


def _parse_frame_id_from_name(name: str) -> tuple[int, str]:
    """Parse ``rgb_<frame_id>.png`` and preserve the original frame-id string."""
    if not name.startswith("rgb_"):
        raise ValueError(f"Expected a rgb_<frame_id> file name, got {name!r}")
    fid_str = name.split("rgb_", 1)[1]
    if "." in fid_str:
        fid_str = fid_str.split(".", 1)[0]
    if not fid_str.isdecimal():
        raise ValueError(f"Frame id must be a decimal integer in file name {name!r}")
    return int(fid_str), fid_str


def parse_frame_cam_from_path(path: Path) -> tuple[int, str, str]:
    """Parse frame id and camera id from ``images/<camera>/rgb/rgb_<id>.png``."""
    fid_int, fid_str = _parse_frame_id_from_name(path.name)
    cam_serial = path.parent.parent.name
    return fid_int, cam_serial, fid_str


def detect_cameras(images_dir: Path) -> list[str]:
    """Return sorted camera folders that contain an ``rgb/`` subfolder."""
    require_dir(images_dir, "scene images directory")
    return [
        cam_dir.name
        for cam_dir in sorted(images_dir.iterdir())
        if cam_dir.is_dir() and (cam_dir / "rgb").is_dir()
    ]


def group_by_frame(images_dir: Path) -> dict[int, list[Path]]:
    """Group camera images by numeric frame id."""
    require_dir(images_dir, "scene images directory")
    grouped: dict[int, list[Path]] = defaultdict(list)

    for image_path in sorted(images_dir.glob("*/*/*.png")):
        if image_path.parent.name != "rgb" or image_path.parent.parent == images_dir:
            continue
        try:
            fid_int, _ = _parse_frame_id_from_name(image_path.name)
        except ValueError:
            continue
        grouped[fid_int].append(image_path)

    out: dict[int, list[Path]] = {}
    for fid, paths in grouped.items():
        out[fid] = sorted(paths, key=lambda p: (p.parent.parent.name, p.name))
    return dict(sorted(out.items()))


def batch_folder_name(index: int) -> str:
    """Return the deterministic 1-based batch folder name."""
    if index < 1:
        raise ValueError("Batch indices are 1-based and must be positive.")
    return f"{index:02d}"


def build_batches(
    frames_sorted: list[int],
    grouped: dict[int, list[Path]],
    batch_size: int,
    overlap: int,
) -> list[list[Path]]:
    """Build overlapping batches from sorted frame ids."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap_frames must be non-negative.")

    step = batch_size - overlap
    if step <= 0:
        raise ValueError("batch_size must be greater than overlap_frames.")

    batches: list[list[Path]] = []
    for start in range(0, len(frames_sorted), step):
        batch_frames = frames_sorted[start : start + batch_size]
        items: list[Path] = []
        for fid in batch_frames:
            items.extend(sorted(grouped[fid], key=lambda p: (p.parent.parent.name, p.name)))
        if items:
            batches.append(items)
    return batches


def parse_batch_range(value: str | None, sep: str = "-") -> tuple[int, int] | None:
    """Parse a 1-based inclusive batch range such as ``3-8`` or ``3:8``."""
    if not value:
        return None
    separators = [sep] if sep in value else ["-", ":"]
    for separator in separators:
        if separator in value:
            left, right = value.split(separator, 1)
            try:
                start, end = int(left), int(right)
            except ValueError as exc:
                raise ValueError(f"Invalid batch range {value!r}") from exc
            if start < 1 or end < start:
                raise ValueError("Batch range must have start >= 1 and end >= start.")
            return start, end
    raise ValueError(f"Invalid batch range {value!r}; expected START-END.")


def write_summary(
    scene_dir: Path,
    cam_serials: list[str],
    expected_per_frame: int,
    frames_sorted: list[int],
    batches_root: Path,
    batch_records: list[dict[str, Any]],
    issues: dict[str, Any],
) -> None:
    """Write JSON and text summaries for the prepared batches."""
    now = datetime.now().isoformat(timespec="seconds")
    summary_json = {
        "generated_at": now,
        "scene_dir": str(scene_dir),
        "images_root": str(scene_dir / "images"),
        "batches_root": str(batches_root),
        "cameras": cam_serials,
        "expected_images_per_frame": expected_per_frame,
        "total_frames_detected": len(frames_sorted),
        "frames_range": [frames_sorted[0], frames_sorted[-1]] if frames_sorted else None,
        "batch_count": len(batch_records),
        "batches": batch_records,
        "issues": issues,
    }

    json_path = scene_dir / "batch_summary.json"
    json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "SDrAwberry Batch Summary",
        "=" * 72,
        f"Generated at    : {now}",
        f"Scene dir       : {scene_dir}",
        f"Images root     : {scene_dir / 'images'}",
        f"Batches root    : {batches_root}",
        f"Cameras ({len(cam_serials)}): {', '.join(cam_serials) if cam_serials else '(none)'}",
        f"Expected/frame  : {expected_per_frame}",
        f"Total frames    : {len(frames_sorted)}",
    ]
    if frames_sorted:
        lines.append(f"Frames range    : {frames_sorted[0]} .. {frames_sorted[-1]}")
    lines.extend(["", f"Total batches   : {len(batch_records)}"])

    for record in batch_records:
        lines.append(
            f"- Batch {record['name']}: {record['image_count']} images, "
            f"frames {record['frame_first']}..{record['frame_last']} "
            f"(frames={len(record['frames'])})"
        )

    lines.append("")
    bad_frames = issues.get("bad_frames", {})
    if bad_frames:
        lines.append(
            f"Issues: {len(bad_frames)} frame(s) did not match the expected "
            f"{expected_per_frame} images:"
        )
        for fid, count in sorted(bad_frames.items())[:20]:
            lines.append(f"  - frame {fid}: {count} image(s)")
    else:
        lines.append("Issues: None")

    txt_path = scene_dir / "batch_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote batch summaries: %s and %s", json_path, txt_path)


def split_into_scenes(
    scene_dir: Path,
    batch_size: int,
    overlap: int,
    symlink: bool = False,
    max_batches: int | None = None,
    cams_per_frame: int | None = None,
    batch_range: tuple[int, int] | None = None,
) -> tuple[Path, int, list[int], list[str], int, list[dict[str, Any]], dict[str, Any]]:
    """Create ``scene_dir/batches/<NN>/images`` folders populated with renamed images."""
    scene_dir = Path(scene_dir)
    images_dir = scene_dir / "images"
    require_dir(images_dir, "input image tree")

    grouped = group_by_frame(images_dir)
    if not grouped:
        raise RuntimeError("No images matched images/<camera_id>/rgb/rgb_<frame_id>.png")

    cam_serials = detect_cameras(images_dir)
    if not cam_serials:
        raise RuntimeError("No camera folders found under images/<camera_id>/rgb/.")
    expected_per_frame = cams_per_frame if cams_per_frame is not None else len(cam_serials)

    bad_frames = {fid: len(paths) for fid, paths in grouped.items() if len(paths) != expected_per_frame}
    if bad_frames:
        preview = ", ".join(f"{fid}:{count}" for fid, count in list(sorted(bad_frames.items()))[:5])
        LOGGER.warning(
            "%d frame(s) do not have exactly %d images. First examples: %s",
            len(bad_frames),
            expected_per_frame,
            preview,
        )
    issues = {"bad_frames": bad_frames}

    frames_sorted = list(grouped.keys())
    batches_all = build_batches(frames_sorted, grouped, batch_size, overlap)
    num_global = len(batches_all)

    if batch_range is not None:
        start, end = batch_range
        if end > num_global:
            raise ValueError(f"batch_range end {end} exceeds total batches {num_global}.")
        selected_indices = list(range(start, end + 1))
    elif max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be >= 1.")
        selected_indices = list(range(1, min(max_batches, num_global) + 1))
    else:
        selected_indices = list(range(1, num_global + 1))

    batches_root = ensure_dir(scene_dir / "batches")

    legacy_conflicts: list[tuple[str, str]] = []
    for index in selected_indices:
        name_2 = batch_folder_name(index)
        name_3 = f"{index:03d}"
        if name_2 != name_3 and (batches_root / name_2).exists() and (batches_root / name_3).exists():
            legacy_conflicts.append((name_2, name_3))
    if legacy_conflicts:
        msg = "\n".join(f"  {a} vs {b}" for a, b in legacy_conflicts[:10])
        raise RuntimeError(
            "Duplicate 2-digit and legacy 3-digit batch folders detected:\n"
            f"{msg}\nResolve the duplicate folders before re-running."
        )

    batch_records: list[dict[str, Any]] = []
    for index in selected_indices:
        batch = batches_all[index - 1]
        batch_name = batch_folder_name(index)

        legacy = batches_root / f"{index:03d}"
        target = batches_root / batch_name
        if legacy.exists() and not target.exists() and legacy != target:
            legacy.rename(target)

        out_images = ensure_dir(target / "images")
        frame_ids_in_batch: list[int] = []

        for src in batch:
            fid_int, cam_serial, fid_str = parse_frame_cam_from_path(src)
            frame_ids_in_batch.append(fid_int)
            dst = out_images / f"{fid_str}_{cam_serial}.png"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        fids_sorted = sorted(set(frame_ids_in_batch))
        LOGGER.info(
            "Prepared batch %s with %d images, frames %s-%s",
            batch_name,
            len(batch),
            fids_sorted[0] if fids_sorted else "n/a",
            fids_sorted[-1] if fids_sorted else "n/a",
        )
        batch_records.append(
            {
                "index": index,
                "name": batch_name,
                "path": str(target),
                "images_dir": str(out_images),
                "image_count": len(batch),
                "frames": fids_sorted,
                "frame_first": fids_sorted[0] if fids_sorted else None,
                "frame_last": fids_sorted[-1] if fids_sorted else None,
            }
        )

    return batches_root, len(selected_indices), frames_sorted, cam_serials, expected_per_frame, batch_records, issues


def _sorted_batch_dirs(batches_root: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.name))
        except ValueError:
            return (1, path.name)

    return sorted([p for p in batches_root.iterdir() if p.is_dir()], key=sort_key)


def _ensure_da3_available() -> None:
    if shutil.which("da3") is None:
        raise FileNotFoundError(
            "DA3 CLI executable 'da3' was not found on PATH. Install Depth-Anything-3 "
            "separately, or run batching with --prepare_only and use existing COLMAP/PLY outputs "
            "for downstream stages."
        )


def run_da3_scene(
    images_dir: Path,
    export_dir: Path,
    model_dir: str,
    device: str,
    conf_thresh_percentile: float,
    num_max_points: int,
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
    export_ply: bool,
    disable_triton: bool,
) -> None:
    """Run DA3 on a batch image folder and export a COLMAP-compatible model."""
    require_dir(images_dir, "DA3 batch images directory")
    _ensure_da3_available()

    if export_dir.exists():
        shutil.rmtree(export_dir)
    ensure_dir(export_dir)

    env = os.environ.copy()
    if disable_triton:
        env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    cmd = [
        "da3",
        "images",
        str(images_dir),
        "--export-format",
        "colmap",
        "--export-dir",
        str(export_dir),
        "--model-dir",
        model_dir,
        "--device",
        device,
        "--conf-thresh-percentile",
        str(conf_thresh_percentile),
        "--num-max-points",
        str(num_max_points),
        "--process-res",
        str(process_res),
        "--process-res-method",
        process_res_method,
        "--auto-cleanup",
        "--ref-view-strategy",
        ref_view_strategy,
    ]
    LOGGER.info("Running DA3: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    if export_ply:
        ply_path = export_dir / "points.ply"
        export_points_ply_from_colmap_model(export_dir, ply_path)
        LOGGER.info("Wrote point cloud: %s", ply_path)


def run_batches(scene_dir: Path, batches_root: Path, config: BatchConfig) -> None:
    """Run DA3 for selected prepared batch folders."""
    sparse_root = ensure_dir(scene_dir / "sparse")
    batch_dirs = _sorted_batch_dirs(batches_root)

    if config.only_batch is not None:
        target = batch_folder_name(config.only_batch)
        batch_dirs = [p for p in batch_dirs if p.name == target]
        if not batch_dirs:
            raise ValueError(f"only_batch {config.only_batch} not found under {batches_root}.")
    elif config.batch_range is not None:
        start, end = config.batch_range
        batch_dirs = [p for p in batch_dirs if p.name.isdecimal() and start <= int(p.name) <= end]
    elif config.max_batches is not None:
        batch_dirs = batch_dirs[: config.max_batches]

    if not batch_dirs:
        raise RuntimeError(f"No prepared batch folders found in {batches_root}.")

    for batch_dir in batch_dirs:
        LOGGER.info("Running batch %s", batch_dir.name)
        run_da3_scene(
            images_dir=batch_dir / "images",
            export_dir=sparse_root / batch_dir.name,
            model_dir=config.da3_model_dir,
            device=config.da3_device,
            conf_thresh_percentile=config.da3_conf_thresh_percentile,
            num_max_points=config.da3_num_max_points,
            process_res=config.da3_process_res,
            process_res_method=config.da3_process_res_method,
            ref_view_strategy=config.da3_ref_view_strategy,
            export_ply=config.da3_export_ply,
            disable_triton=config.da3_disable_triton,
        )


def prepare_and_run(config: BatchConfig) -> None:
    """Prepare batches, write summaries, and optionally run DA3."""
    scene_dir = Path(config.scene_dir).resolve()
    selected_max = config.max_batches if (config.only_batch is None and config.batch_range is None) else None

    (
        batches_root,
        total_prepared,
        frames_sorted,
        cam_serials,
        expected_per_frame,
        batch_records,
        issues,
    ) = split_into_scenes(
        scene_dir=scene_dir,
        batch_size=config.batch_size,
        overlap=config.overlap_frames,
        symlink=config.symlink,
        max_batches=selected_max,
        cams_per_frame=config.cams_per_frame,
        batch_range=config.batch_range,
    )
    LOGGER.info("Total batches prepared this run: %d", total_prepared)

    write_summary(
        scene_dir=scene_dir,
        cam_serials=cam_serials,
        expected_per_frame=expected_per_frame,
        frames_sorted=frames_sorted,
        batches_root=batches_root,
        batch_records=batch_records,
        issues=issues,
    )

    if config.prepare_only:
        LOGGER.info("prepare_only=True; skipping DA3 execution.")
        return

    run_batches(scene_dir=scene_dir, batches_root=batches_root, config=config)


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the batching CLI parser."""
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Prepare overlapping multi-camera batches and optionally run DA3.")
    ap.add_argument("--scene_dir", required="scene_dir" not in defaults, default=defaults.get("scene_dir"))
    ap.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 10))
    ap.add_argument("--overlap_frames", type=int, default=defaults.get("overlap_frames", 0))
    ap.add_argument("--symlink", action="store_true", default=defaults.get("symlink", False))
    ap.add_argument("--only_batch", type=int, default=defaults.get("only_batch"))
    ap.add_argument("--max_batches", type=int, default=defaults.get("max_batches"))
    ap.add_argument("--cams_per_frame", type=int, default=defaults.get("cams_per_frame"))
    ap.add_argument("--batch_range", type=str, default=defaults.get("batch_range"))
    ap.add_argument("--prepare_only", action="store_true", default=defaults.get("prepare_only", False))
    ap.add_argument("--da3_model_dir", default=defaults.get("da3_model_dir", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"))
    ap.add_argument("--da3_device", default=defaults.get("da3_device", "cuda"))
    ap.add_argument("--da3_conf_thresh_percentile", type=float, default=defaults.get("da3_conf_thresh_percentile", 75.0))
    ap.add_argument("--da3_num_max_points", type=int, default=defaults.get("da3_num_max_points", 1_000_000))
    ap.add_argument("--da3_process_res", type=int, default=defaults.get("da3_process_res", 504))
    ap.add_argument("--da3_process_res_method", default=defaults.get("da3_process_res_method", "upper_bound_resize"))
    ap.add_argument("--da3_ref_view_strategy", default=defaults.get("da3_ref_view_strategy", "saddle_balanced"))
    ap.add_argument("--da3_export_ply", action="store_true", default=defaults.get("da3_export_ply", False))
    ap.add_argument("--da3_disable_triton", action="store_true", default=defaults.get("da3_disable_triton", False))
    ap.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    return ap


def config_from_args(args: argparse.Namespace) -> BatchConfig:
    """Convert parsed CLI args into a ``BatchConfig``."""
    if args.batch_range and args.only_batch is not None:
        raise SystemExit("--batch_range and --only_batch are mutually exclusive.")
    if args.batch_range and args.max_batches is not None:
        raise SystemExit("--batch_range and --max_batches are mutually exclusive.")

    batch_range = parse_batch_range(args.batch_range) if args.batch_range else None
    return BatchConfig(
        scene_dir=Path(args.scene_dir),
        batch_size=int(args.batch_size),
        overlap_frames=int(args.overlap_frames),
        symlink=bool(args.symlink),
        only_batch=args.only_batch,
        max_batches=args.max_batches,
        cams_per_frame=args.cams_per_frame,
        batch_range=batch_range,
        prepare_only=bool(args.prepare_only),
        da3_model_dir=str(args.da3_model_dir),
        da3_device=str(args.da3_device),
        da3_conf_thresh_percentile=float(args.da3_conf_thresh_percentile),
        da3_num_max_points=int(args.da3_num_max_points),
        da3_process_res=int(args.da3_process_res),
        da3_process_res_method=str(args.da3_process_res_method),
        da3_ref_view_strategy=str(args.da3_ref_view_strategy),
        da3_export_ply=bool(args.da3_export_ply),
        da3_disable_triton=bool(args.da3_disable_triton),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``sdrawberry-batch``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(args.verbose))
    prepare_and_run(config_from_args(args))


if __name__ == "__main__":
    main()

"""Streamed fusion of per-batch COLMAP/PLY reconstructions."""

from __future__ import annotations

import argparse
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .colmap_io import load_views_and_points_from_colmap, model_files_exist
from .io_utils import ensure_dir, require_dir, setup_logging

LOGGER = logging.getLogger(__name__)
DTYPE = np.float32


@dataclass
class FusionConfig:
    """Configuration for streamed global point-cloud fusion."""

    base_dir: Path
    sparse_subfolder: str = "sparse"
    ply: str = "points.ply"
    out: Path = Path("merged/merged_global.ply")
    max_batches: int | None = None
    batch_range: tuple[int, int] | None = None
    model_ext: str = ""
    icp_iters: int = 20
    window_kdt: int = 4
    window_match: int = 2
    voxel: float = 0.002
    pre_voxel: float = 0.003
    geom_radius: float = 0.0015
    match_radius: float = 0.0008
    k_for_svd: int = 5000


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for point-cloud fusion.") from exc
    return o3d


def parse_range(value: str | None) -> tuple[int, int] | None:
    """Parse a numeric batch range ``START:END`` or ``START-END``."""
    if not value:
        return None
    separator = ":" if ":" in value else "-"
    try:
        start_str, end_str = value.split(separator, 1)
        start, end = int(start_str), int(end_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError("batch_range must be START:END, e.g. 15:75") from exc
    if start > end:
        start, end = end, start
    return start, end


def sorted_batches(base_dir: Path, sparse_subfolder: str) -> list[tuple[int | None, Path]]:
    """List COLMAP sparse subfolders in numeric order when possible."""
    sparse_root = Path(base_dir) / sparse_subfolder
    require_dir(sparse_root, "COLMAP sparse root")

    folders = [p for p in sparse_root.iterdir() if p.is_dir()]

    def to_int(path: Path) -> int | None:
        try:
            return int(path.name)
        except ValueError:
            return None

    infos = [(to_int(path), path) for path in folders]
    if any(index is not None for index, _ in infos):
        numeric = sorted([item for item in infos if item[0] is not None], key=lambda item: int(item[0]))
        text = sorted([item for item in infos if item[0] is None], key=lambda item: item[1].name)
        return numeric + text
    return sorted(infos, key=lambda item: item[1].name)


def select_batches(
    all_infos: list[tuple[int | None, Path]],
    batch_range: tuple[int, int] | None = None,
    max_batches: int | None = None,
) -> list[tuple[int | None, Path]]:
    """Select batch folders based on numeric range and/or maximum count."""
    if batch_range is None:
        selected = list(all_infos)
    else:
        start, end = batch_range
        selected = [(index, path) for index, path in all_infos if index is not None and start <= index <= end]

    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be >= 1.")
        selected = selected[:max_batches]
    return selected


def find_matches_from_views(
    view_index_1: dict[tuple[str, tuple[float, float]], list[int]],
    view_index_2: dict[tuple[str, tuple[float, float]], list[int]],
    points1: dict[int, np.ndarray],
    points2: dict[int, np.ndarray],
    k: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D-3D correspondences from shared COLMAP image-feature tracks."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    matched_b: list[np.ndarray] = []

    for view, ids2_all in view_index_2.items():
        if view not in view_index_1:
            continue
        ids1 = view_index_1[view][:k]
        ids2 = ids2_all[:k]
        for id1 in ids1:
            for id2 in ids2:
                if id1 in points1 and id2 in points2:
                    pairs.append((points1[id1], points2[id2]))
        matched_b.extend([points2[idx] for idx in ids2_all if idx in points2])

    if not pairs:
        x1 = np.zeros((0, 3), dtype=np.float64)
        x2 = np.zeros((0, 3), dtype=np.float64)
    else:
        x1, x2 = map(np.asarray, zip(*pairs))

    mb = np.unique(np.asarray(matched_b), axis=0) if matched_b else np.zeros((0, 3), dtype=np.float64)
    return np.asarray(x1, dtype=np.float64), np.asarray(x2, dtype=np.float64), mb


def estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    force_unit_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate a similarity transform ``target ~= s * R * source + t``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape (N, 3).")
    if len(source) < 3:
        raise ValueError("At least three points are required to estimate a 3D transform.")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    x = source - mu_source
    y = target - mu_target

    sigma = (y.T @ x) / max(1, len(source))
    u, singular_values, vt = np.linalg.svd(sigma)

    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1

    rotation = u @ sign @ vt
    var_source = (x**2).sum() / max(1, len(source))
    scale = 1.0 if force_unit_scale else float(np.trace(np.diag(singular_values) @ sign) / (var_source + 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return float(scale), rotation, translation


def roll_pairs_refined(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    t_init: np.ndarray | None = None,
    max_iter: int = 3,
) -> np.ndarray:
    """Optimize single-axis rotation around reconstruction Y plus translation."""
    x_src = np.asarray(x_src, dtype=np.float64)
    x_tgt = np.asarray(x_tgt, dtype=np.float64)
    if len(x_src) < 3 or len(x_src) != len(x_tgt):
        return np.eye(4, dtype=np.float64)

    if t_init is None:
        _, rotation0, translation0 = estimate_similarity_transform(x_src, x_tgt, force_unit_scale=True)
    else:
        rotation0 = t_init[:3, :3]
        translation0 = t_init[:3, 3]

    roll = math.atan2(rotation0[0, 2], rotation0[0, 0])
    c, s = math.cos(roll), math.sin(roll)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    translation = translation0.copy()

    mu_src = x_src.mean(axis=0)
    mu_tgt = x_tgt.mean(axis=0)

    for _ in range(max(1, int(max_iter))):
        x_centered = x_src - mu_src
        y_centered = x_tgt - mu_tgt
        h = y_centered.T @ x_centered
        denom = h[0, 0] + h[2, 2]
        if abs(denom) < 1e-12:
            denom = 1e-12
        roll = math.atan2(h[0, 2] - h[2, 0], denom)
        c, s = math.cos(roll), math.sin(roll)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        translation = mu_tgt - rotation @ mu_src

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _intvector_to_list(index_vector: Any) -> list[int]:
    try:
        return list(index_vector)
    except TypeError:
        return np.asarray(index_vector, dtype=np.int64).tolist()


def to_f32_pts(values: np.ndarray) -> np.ndarray:
    """Convert an array to float32 without unnecessary copies."""
    return np.asarray(values).astype(DTYPE, copy=False)


class VoxelHash:
    """Voxel-hash accumulator that exports centroid points with average colors."""

    def __init__(self, voxel_size: float):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        self.voxel_size = float(voxel_size)
        self.map: dict[tuple[int, int, int], tuple[float, float, float, float, float, float, int]] = {}
        self.has_color = False

    def add_points(self, points: np.ndarray, colors: np.ndarray | None = None) -> None:
        """Accumulate points and optional RGB colors in ``[0, 1]``."""
        points = np.asarray(points)
        has_color = colors is not None and len(colors) == len(points)
        colors_arr = np.asarray(colors, dtype=np.float32, order="C") if has_color else None
        self.has_color = self.has_color or has_color

        for i, point in enumerate(points):
            if not np.all(np.isfinite(point)):
                continue
            key = (
                int(math.floor(float(point[0]) / self.voxel_size)),
                int(math.floor(float(point[1]) / self.voxel_size)),
                int(math.floor(float(point[2]) / self.voxel_size)),
            )
            if key in self.map:
                sx, sy, sz, cr, cg, cb, n = self.map[key]
                sx += float(point[0])
                sy += float(point[1])
                sz += float(point[2])
                n += 1
                if colors_arr is not None:
                    r, g, b = colors_arr[i]
                    cr += float(r)
                    cg += float(g)
                    cb += float(b)
                self.map[key] = (sx, sy, sz, cr, cg, cb, n)
            elif colors_arr is not None:
                r, g, b = colors_arr[i]
                self.map[key] = (float(point[0]), float(point[1]), float(point[2]), float(r), float(g), float(b), 1)
            else:
                self.map[key] = (float(point[0]), float(point[1]), float(point[2]), 0.0, 0.0, 0.0, 1)

    def to_pointcloud(self) -> Any:
        """Convert accumulated voxels into an Open3D point cloud."""
        o3d = _require_open3d()
        nvox = len(self.map)
        points = np.empty((nvox, 3), dtype=np.float32)
        colors = np.empty((nvox, 3), dtype=np.float32)

        for i, (sx, sy, sz, cr, cg, cb, n) in enumerate(self.map.values()):
            inv = 1.0 / n
            points[i] = (sx * inv, sy * inv, sz * inv)
            if self.has_color:
                colors[i] = (cr * inv, cg * inv, cb * inv)
            else:
                colors[i] = (0.0, 0.0, 0.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if self.has_color:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
        return pcd


def build_kdt(pcd: Any) -> Any | None:
    """Build an Open3D KD-tree for a non-empty point cloud."""
    if len(pcd.points) == 0:
        return None
    o3d = _require_open3d()
    return o3d.geometry.KDTreeFlann(pcd)


def remove_by_match_seed_local(
    pcd_local: Any,
    views_cur: dict[tuple[str, tuple[float, float]], list[int]] | None,
    pts3d_cur: dict[int, np.ndarray] | None,
    recent_parents: list[Path],
    k_for_svd: int,
    radius: float,
    model_ext: str = "",
) -> tuple[Any, int]:
    """Remove local points close to current-batch COLMAP match seeds."""
    if len(pcd_local.points) == 0 or views_cur is None or pts3d_cur is None:
        return pcd_local, 0

    o3d = _require_open3d()
    remove_indices: set[int] = set()
    kdt_local = o3d.geometry.KDTreeFlann(pcd_local)

    for parent_path in recent_parents:
        pts3d_parent, views_parent = load_views_and_points_from_colmap(parent_path, ext=model_ext)
        _, _, matched_b = find_matches_from_views(views_parent, views_cur, pts3d_parent, pts3d_cur, k=k_for_svd)
        for query in matched_b:
            count, indices, _ = kdt_local.search_radius_vector_3d(np.asarray(query, dtype=np.float64), radius)
            if count > 0:
                remove_indices.update(_intvector_to_list(indices))

    if not remove_indices:
        return pcd_local, 0

    keep = np.setdiff1d(np.arange(len(pcd_local.points)), np.fromiter(remove_indices, dtype=int))
    return pcd_local.select_by_index(keep), len(remove_indices)


def _validate_selected_batches(selected_infos: list[tuple[int | None, Path]], ply_name: str, need_colmap: bool) -> None:
    for index, path in selected_infos:
        ply_path = path / ply_name
        if not ply_path.is_file():
            raise FileNotFoundError(f"Missing per-batch point cloud for batch {index or path.name}: {ply_path}")
        if need_colmap and not model_files_exist(path):
            raise FileNotFoundError(
                f"Missing COLMAP sparse model for batch {index or path.name}: {path}. "
                "Fusion needs cameras/images/points3D files for cross-batch correspondences."
            )


def run_fusion(config: FusionConfig) -> Path:
    """Run streamed fusion and return the output PLY path."""
    o3d = _require_open3d()
    base_dir = Path(config.base_dir).resolve()
    all_infos = sorted_batches(base_dir, config.sparse_subfolder)
    if not all_infos:
        raise RuntimeError(f"No batch folders found under {base_dir / config.sparse_subfolder}.")

    selected_infos = select_batches(all_infos, batch_range=config.batch_range, max_batches=config.max_batches)
    if not selected_infos:
        raise RuntimeError("No batches selected. Check batch_range and max_batches.")

    _validate_selected_batches(selected_infos, config.ply, need_colmap=len(selected_infos) > 1)
    LOGGER.info("Selected %d batch(es) for fusion.", len(selected_infos))

    voxels = VoxelHash(config.voxel)
    window_global_pcds: deque[Any] = deque()
    window_kdtrees: deque[Any] = deque()
    global_poses: list[np.ndarray] = []

    for idx, (_batch_index, path) in enumerate(selected_infos):
        LOGGER.info("Processing batch %d/%d: %s", idx + 1, len(selected_infos), path)
        pcd_path = path / config.ply
        p_local = o3d.io.read_point_cloud(str(pcd_path))
        if p_local.is_empty():
            raise ValueError(f"Empty point cloud: {pcd_path}")
        if config.pre_voxel and config.pre_voxel > 0:
            p_local = p_local.voxel_down_sample(float(config.pre_voxel))
            if p_local.is_empty():
                raise ValueError(f"Point cloud became empty after pre_voxel downsampling: {pcd_path}")

        views_cur = pts3d_cur = None
        if idx > 0:
            pts3d_cur, views_cur = load_views_and_points_from_colmap(path, ext=config.model_ext)

        removed_by_match = 0
        if idx > 0 and config.window_match > 0:
            parent_paths = [selected_infos[j][1] for j in range(max(0, idx - config.window_match), idx)]
            p_local, removed_by_match = remove_by_match_seed_local(
                p_local,
                views_cur,
                pts3d_cur,
                parent_paths,
                config.k_for_svd,
                config.match_radius,
                model_ext=config.model_ext,
            )
        LOGGER.info("Removed %d local points by match-seed dedup.", removed_by_match)

        t_rel = np.eye(4, dtype=np.float64)
        parent_found = False
        parent_idx_for_pose = idx - 1

        if idx > 0:
            for back in (1, 2):
                parent_idx = idx - back
                if parent_idx < 0:
                    continue
                parent_path = selected_infos[parent_idx][1]
                pts3d_parent, views_parent = load_views_and_points_from_colmap(parent_path, ext=config.model_ext)
                x_parent, x_current, _ = find_matches_from_views(
                    views_parent,
                    views_cur or {},
                    pts3d_parent,
                    pts3d_cur or {},
                    k=config.k_for_svd,
                )
                if len(x_parent) < 3:
                    continue

                _, rotation0, translation0 = estimate_similarity_transform(x_current, x_parent, force_unit_scale=True)
                t_init = np.eye(4, dtype=np.float64)
                t_init[:3, :3] = rotation0
                t_init[:3, 3] = translation0

                t_rel = roll_pairs_refined(
                    x_src=x_current,
                    x_tgt=x_parent,
                    t_init=t_init,
                    max_iter=max(1, config.icp_iters),
                )
                x_current_to_parent = (t_rel[:3, :3] @ x_current.T).T + t_rel[:3, 3]
                rmse = float(np.sqrt(np.mean(np.sum((x_current_to_parent - x_parent) ** 2, axis=1))))
                roll_deg = math.degrees(math.atan2(t_rel[0, 2], t_rel[0, 0]))
                LOGGER.info(
                    "Roll-constrained pose: parent=%d pairs=%d roll=%.3f deg rmse=%.6f",
                    parent_idx,
                    len(x_parent),
                    roll_deg,
                    rmse,
                )
                parent_found = True
                parent_idx_for_pose = parent_idx
                break

        if idx == 0:
            global_current = np.eye(4, dtype=np.float64)
        elif parent_found:
            global_current = global_poses[parent_idx_for_pose] @ t_rel
        else:
            LOGGER.warning("No COLMAP correspondences found for batch %d; using identity relative pose.", idx)
            global_current = global_poses[-1].copy()
        global_poses.append(global_current)

        local_points = to_f32_pts(np.asarray(p_local.points))
        local_colors = None
        if len(p_local.colors) == len(p_local.points) and len(p_local.colors) > 0:
            local_colors = np.asarray(p_local.colors, dtype=np.float32)

        if local_points.size == 0:
            raise ValueError(f"No local points remain for batch {path}.")

        global_points = (global_current[:3, :3] @ local_points.T).T + global_current[:3, 3]
        global_colors = local_colors

        if len(window_kdtrees) > 0:
            keep = np.ones(len(global_points), dtype=bool)
            for kdt in reversed(window_kdtrees):
                if kdt is None:
                    continue
                for point_index, query in enumerate(global_points):
                    if not keep[point_index]:
                        continue
                    count, _, _ = kdt.search_radius_vector_3d(query.astype(np.float64), float(config.geom_radius))
                    if count > 0:
                        keep[point_index] = False
            global_points = global_points[keep]
            if global_colors is not None:
                global_colors = global_colors[keep]

        LOGGER.info("Kept %d global points after sliding-window geometry dedup.", len(global_points))
        voxels.add_points(global_points, global_colors)

        pcd_window = o3d.geometry.PointCloud()
        pcd_window.points = o3d.utility.Vector3dVector(global_points.astype(np.float64))
        if global_colors is not None:
            pcd_window.colors = o3d.utility.Vector3dVector(np.clip(global_colors, 0.0, 1.0).astype(np.float64))

        window_global_pcds.append(pcd_window)
        window_kdtrees.append(build_kdt(pcd_window))
        while len(window_global_pcds) > max(0, int(config.window_kdt)):
            window_global_pcds.popleft()
            window_kdtrees.popleft()

    merged = voxels.to_pointcloud()
    if merged.is_empty():
        raise RuntimeError("Fusion produced an empty point cloud.")

    out_path = Path(config.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    ensure_dir(out_path.parent)
    o3d.io.write_point_cloud(str(out_path), merged)
    LOGGER.info("Saved merged point cloud to %s (#voxels=%d)", out_path, len(voxels.map))
    return out_path


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the fusion CLI parser."""
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Fuse per-batch COLMAP/PLY reconstructions into one point cloud.")
    ap.add_argument("--base_dir", required="base_dir" not in defaults, default=defaults.get("base_dir"))
    ap.add_argument("--sparse_subfolder", default=defaults.get("sparse_subfolder", "sparse"))
    ap.add_argument("--ply", default=defaults.get("ply", "points.ply"))
    ap.add_argument("--out", default=defaults.get("out", "merged/merged_global.ply"))
    ap.add_argument("--max_batches", type=int, default=defaults.get("max_batches"))
    ap.add_argument("--batch_range", type=str, default=defaults.get("batch_range"))
    ap.add_argument("--model_ext", default=defaults.get("model_ext", ""), choices=["", ".txt", ".bin"])
    ap.add_argument("--icp_iters", type=int, default=defaults.get("icp_iters", 20))
    ap.add_argument("--window_kdt", type=int, default=defaults.get("window_kdt", 4))
    ap.add_argument("--window_match", type=int, default=defaults.get("window_match", 2))
    ap.add_argument("--voxel", type=float, default=defaults.get("voxel", 0.002))
    ap.add_argument("--pre_voxel", type=float, default=defaults.get("pre_voxel", 0.003))
    ap.add_argument("--geom_radius", type=float, default=defaults.get("geom_radius", 0.0015))
    ap.add_argument("--match_radius", type=float, default=defaults.get("match_radius", 0.0008))
    ap.add_argument("--k_for_svd", type=int, default=defaults.get("k_for_svd", 5000))
    ap.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    return ap


def config_from_args(args: argparse.Namespace) -> FusionConfig:
    """Convert parsed args to ``FusionConfig``."""
    return FusionConfig(
        base_dir=Path(args.base_dir),
        sparse_subfolder=str(args.sparse_subfolder),
        ply=str(args.ply),
        out=Path(args.out),
        max_batches=args.max_batches,
        batch_range=parse_range(args.batch_range),
        model_ext=str(args.model_ext),
        icp_iters=int(args.icp_iters),
        window_kdt=int(args.window_kdt),
        window_match=int(args.window_match),
        voxel=float(args.voxel),
        pre_voxel=float(args.pre_voxel),
        geom_radius=float(args.geom_radius),
        match_radius=float(args.match_radius),
        k_for_svd=int(args.k_for_svd),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``sdrawberry-fusion``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(args.verbose))
    run_fusion(config_from_args(args))


if __name__ == "__main__":
    main()

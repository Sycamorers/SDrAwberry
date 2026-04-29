"""Plant-level point-cloud extraction from a fused reconstruction."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for support-plane, HSV, DBSCAN, and ground-like filtering."""

    input_pcd: Path
    out_dir: Path
    use_auto_scale: bool = False
    z_axis_inverted: bool = False
    plane_dist_threshold: float = 0.005
    plane_dist_threshold_scale: float = 0.005
    plane_ransac_n: int = 3
    plane_iter: int = 1000
    use_nonplant_height_cut: bool = False
    auto_height_margin_factor: float = 4.0
    leaf_h_min: int = 30
    leaf_h_max: int = 95
    leaf_s_min: int = 60
    leaf_v_min: int = 40
    stem_red_1_max: int = 20
    stem_red_2_min: int = 160
    stem_yellow_min: int = 20
    stem_yellow_max: int = 40
    stem_s_min: int = 40
    stem_v_min: int = 40
    dbscan_eps: float = 0.015
    dbscan_eps_scale: float = 0.02
    dbscan_min_pts: int = 120
    save_clustered_pcd: bool = False
    clustered_output_path: Path | None = None
    save_each_cluster_separately: bool = True
    cluster_min_size: int | None = 10000
    cluster_max_size: int | None = None
    save_vis_all_colored: bool = False
    vis_all_colored_pcd_path: Path | None = None
    save_vis_clusters_colored: bool = False
    vis_clusters_colored_pcd_path: Path | None = None
    debug_save_cluster_stats: bool = False
    cluster_stats_path: Path | None = None
    cluster_plot_path: Path | None = None
    filter_groundlike_clusters: bool = True
    ground_median_margin_factor: float = 3.0
    ground_height_range_factor: float = 4.0
    ground_planarity_ratio_max: float = 0.02
    use_nonplant_for_plane_fit: bool = False
    visualize: bool = False


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for clustering point clouds.") from exc
    return o3d


def normalize_plane(plane_model: tuple[float, float, float, float] | list[float] | np.ndarray) -> tuple[float, float, float, float]:
    """Normalize a plane equation ``ax + by + cz + d = 0``."""
    a, b, c, d = [float(v) for v in plane_model]
    norm = float(np.linalg.norm([a, b, c]))
    if norm < 1e-12:
        raise ValueError("Plane normal norm is too small; plane fitting failed.")
    return a / norm, b / norm, c / norm, d / norm


def signed_plane_distances(
    points: np.ndarray,
    plane_model: tuple[float, float, float, float],
    z_axis_inverted: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Compute signed plane-relative distances, optionally flipping plant direction."""
    a, b, c, d = normalize_plane(plane_model)
    distances = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    if z_axis_inverted:
        distances = -distances
        a, b, c, d = -a, -b, -c, -d
    return distances, (a, b, c, d)


def cluster_planarity_ratio(points: np.ndarray) -> float:
    """Return the PCA small-to-large eigenvalue ratio for cluster planarity."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(covariance)
    return float(eigvals[0] / (eigvals[2] + 1e-12))


def is_groundlike_cluster(
    points: np.ndarray,
    distances: np.ndarray,
    ground_ref: float,
    plane_dist_threshold: float,
    median_margin_factor: float,
    height_range_factor: float,
    planarity_ratio_max: float,
) -> bool:
    """Classify a cluster as ground-like using height and planarity tests."""
    p5, p95 = np.percentile(distances, [5, 95])
    height_range = float(p95 - p5)
    median_height = float(np.median(distances))
    planar_ratio = cluster_planarity_ratio(points)
    return bool(
        (median_height < ground_ref + median_margin_factor * plane_dist_threshold)
        and (height_range < height_range_factor * plane_dist_threshold)
        and (planar_ratio < planarity_ratio_max)
    )


def hsv_plant_mask(colors: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return plant, leaf, and stem HSV masks for RGB colors in ``[0, 1]``."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for HSV-guided plant extraction.") from exc

    if colors.size == 0:
        raise ValueError("Point cloud has no colors; HSV plant extraction requires RGB colors.")

    rgb_255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_255.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    leaf = (
        (h > config.leaf_h_min)
        & (h < config.leaf_h_max)
        & (s > config.leaf_s_min)
        & (v > config.leaf_v_min)
    )
    stem_red = (h < config.stem_red_1_max) | (h > config.stem_red_2_min)
    stem_yellow = (h > config.stem_yellow_min) & (h < config.stem_yellow_max)
    stem = (stem_red | stem_yellow) & (s > config.stem_s_min) & (v > config.stem_v_min)
    plant = leaf | stem
    return plant, leaf, stem


def _write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    o3d = _require_open3d()
    ensure_dir(path.parent)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(str(path), pcd)


def _optional_path(path: Path | None, default: Path) -> Path:
    return path if path is not None else default


def run_clustering(config: ClusteringConfig) -> dict[str, Any]:
    """Run plant extraction and DBSCAN clustering. Returns a summary dictionary."""
    o3d = _require_open3d()
    input_pcd = Path(config.input_pcd)
    if not input_pcd.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {input_pcd}")

    out_dir = ensure_dir(Path(config.out_dir))
    pcd = o3d.io.read_point_cloud(str(input_pcd))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {input_pcd}")
    if not pcd.has_colors():
        raise ValueError("Input point cloud has no RGB colors; clustering requires colors for HSV filtering.")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    n_points = int(points.shape[0])
    if n_points < int(config.plane_ransac_n):
        raise ValueError(f"Too few points ({n_points}) for RANSAC plane fitting.")

    if config.use_auto_scale:
        scene_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        dbscan_eps = scene_diag * float(config.dbscan_eps_scale)
        plane_dist_threshold = scene_diag * float(config.plane_dist_threshold_scale)
    else:
        scene_diag = None
        dbscan_eps = float(config.dbscan_eps)
        plane_dist_threshold = float(config.plane_dist_threshold)

    if config.use_nonplant_for_plane_fit:
        LOGGER.warning(
            "use_nonplant_for_plane_fit=True is ignored because support-plane estimation is run before HSV filtering."
        )

    plane_model_raw, plane_inliers_local = pcd.segment_plane(
        distance_threshold=plane_dist_threshold,
        ransac_n=int(config.plane_ransac_n),
        num_iterations=int(config.plane_iter),
    )
    plane_inliers = np.asarray(plane_inliers_local, dtype=int)
    distances, plane_model = signed_plane_distances(points, plane_model_raw, z_axis_inverted=config.z_axis_inverted)
    LOGGER.info("Support plane: %.6fx + %.6fy + %.6fz + %.6f = 0", *plane_model)
    LOGGER.info("Plane inliers: %d", int(len(plane_inliers)))

    plant_mask_all, leaf_mask_all, stem_mask_all = hsv_plant_mask(colors, config)
    if int(plant_mask_all.sum()) == 0:
        raise ValueError("HSV filtering did not detect plant-colored points. Check thresholds.")

    keep_mask = np.ones(n_points, dtype=bool)
    if config.use_nonplant_height_cut:
        inlier_mask = np.zeros(n_points, dtype=bool)
        inlier_mask[plane_inliers] = True
        nonplant_mask = ~plant_mask_all
        ground_mask_nonplant = inlier_mask & nonplant_mask
        ground_dists = distances[ground_mask_nonplant]
        if ground_dists.size > 0:
            ground_top = np.percentile(ground_dists, 95)
            height_thresh = ground_top + float(config.auto_height_margin_factor) * plane_dist_threshold
            cut_mask = nonplant_mask & (distances < height_thresh)
            keep_mask[cut_mask] = False
            LOGGER.info("Non-plant height cut removed %d points.", int(cut_mask.sum()))
        else:
            LOGGER.warning("No non-plant plane inliers found; skipping non-plant height cut.")

    final_plant_mask = plant_mask_all & keep_mask
    final_idx = np.where(final_plant_mask)[0]
    if final_idx.size == 0:
        raise ValueError("No plant points remained after HSV filtering.")

    pcd_high = pcd.select_by_index(final_idx)
    labels_high = np.asarray(
        pcd_high.cluster_dbscan(
            eps=dbscan_eps,
            min_points=int(config.dbscan_min_pts),
            print_progress=False,
        )
    )
    max_label = int(labels_high.max()) if labels_high.size else -1
    if max_label < 0:
        raise ValueError("DBSCAN did not produce clusters. Check dbscan_eps and dbscan_min_pts.")

    cluster_labels_pos = labels_high[labels_high >= 0]
    cluster_sizes = np.bincount(cluster_labels_pos)
    LOGGER.info("DBSCAN produced %d cluster label(s).", len(cluster_sizes))

    groundlike_mask = np.zeros_like(cluster_sizes, dtype=bool)
    if config.filter_groundlike_clusters:
        inlier_mask = np.zeros(n_points, dtype=bool)
        inlier_mask[plane_inliers] = True
        ref_dists = distances[inlier_mask]
        ground_ref = float(np.percentile(ref_dists, 95)) if ref_dists.size else float(np.percentile(distances, 5))
        for cid in range(len(cluster_sizes)):
            local_mask = labels_high == cid
            if not np.any(local_mask):
                continue
            global_ids = final_idx[local_mask]
            groundlike_mask[cid] = is_groundlike_cluster(
                points=points[global_ids],
                distances=distances[global_ids],
                ground_ref=ground_ref,
                plane_dist_threshold=plane_dist_threshold,
                median_margin_factor=float(config.ground_median_margin_factor),
                height_range_factor=float(config.ground_height_range_factor),
                planarity_ratio_max=float(config.ground_planarity_ratio_max),
            )
        LOGGER.info("Ground-like clusters removed: %d", int(groundlike_mask.sum()))
    else:
        ground_ref = None

    if config.debug_save_cluster_stats:
        stats_path = _optional_path(config.cluster_stats_path, out_dir / "cluster_size_stats.csv")
        stats_array = np.column_stack((np.arange(len(cluster_sizes), dtype=int), cluster_sizes))
        np.savetxt(stats_path, stats_array, fmt="%d", delimiter=",", header="cluster_id,point_count", comments="")
        plot_path = _optional_path(config.cluster_plot_path, out_dir / "cluster_size_stats.png")
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(len(cluster_sizes), dtype=int), cluster_sizes, width=0.8)
            plt.xlabel("Cluster ID")
            plt.ylabel("Point Count")
            plt.title("Distribution of Cluster Sizes")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
        except ImportError:
            LOGGER.warning("matplotlib is not installed; skipped cluster size plot.")

    valid_mask = np.ones_like(cluster_sizes, dtype=bool)
    if config.filter_groundlike_clusters:
        valid_mask &= ~groundlike_mask
    if config.cluster_min_size is not None:
        valid_mask &= cluster_sizes >= int(config.cluster_min_size)
    if config.cluster_max_size is not None:
        valid_mask &= cluster_sizes <= int(config.cluster_max_size)

    cluster_ids = np.where(valid_mask)[0]
    if cluster_ids.size == 0:
        raise ValueError("All clusters were filtered out. Check size and ground-like thresholds.")

    cluster_assign_global = -1 * np.ones(n_points, dtype=int)
    for local_index, cid in enumerate(labels_high):
        if cid >= 0 and valid_mask[cid]:
            cluster_assign_global[final_idx[local_index]] = int(cid)

    plant_cluster_mask_global = cluster_assign_global >= 0
    cluster_points = points[plant_cluster_mask_global]
    cluster_colors = colors[plant_cluster_mask_global]

    output_clusters: list[dict[str, Any]] = []
    if config.save_clustered_pcd:
        clustered_path = _optional_path(config.clustered_output_path, out_dir / "merged_clustered.ply")
        _write_point_cloud(clustered_path, cluster_points, cluster_colors)

    if config.save_each_cluster_separately:
        for output_index, cid in enumerate(cluster_ids, start=1):
            mask = cluster_assign_global == cid
            out_path = out_dir / f"cluster_{output_index:04d}.ply"
            _write_point_cloud(out_path, points[mask], colors[mask])
            output_clusters.append(
                {
                    "output_index": int(output_index),
                    "source_cluster_id": int(cid),
                    "point_count": int(mask.sum()),
                    "path": str(out_path),
                }
            )

    rng = np.random.default_rng(42)
    palette = rng.random((cluster_ids.size, 3))

    vis_pcd_all = None
    vis_pcd_clusters_colored = None
    if config.save_vis_all_colored or config.save_vis_clusters_colored or config.visualize:
        vis_colors_all = np.full_like(colors, 0.6)
        for palette_index, cid in enumerate(cluster_ids):
            vis_colors_all[cluster_assign_global == cid] = palette[palette_index]

        vis_pcd_all = o3d.geometry.PointCloud()
        vis_pcd_all.points = o3d.utility.Vector3dVector(points)
        vis_pcd_all.colors = o3d.utility.Vector3dVector(vis_colors_all)

        vis_cluster_colors = np.zeros_like(cluster_colors)
        cluster_ids_for_points = cluster_assign_global[plant_cluster_mask_global]
        for palette_index, cid in enumerate(cluster_ids):
            vis_cluster_colors[cluster_ids_for_points == cid] = palette[palette_index]

        vis_pcd_clusters_colored = o3d.geometry.PointCloud()
        vis_pcd_clusters_colored.points = o3d.utility.Vector3dVector(cluster_points)
        vis_pcd_clusters_colored.colors = o3d.utility.Vector3dVector(vis_cluster_colors)

    if config.save_vis_all_colored and vis_pcd_all is not None:
        vis_path = _optional_path(config.vis_all_colored_pcd_path, out_dir / "merged_vis_all_colored.ply")
        o3d.io.write_point_cloud(str(vis_path), vis_pcd_all)
    if config.save_vis_clusters_colored and vis_pcd_clusters_colored is not None:
        vis_clusters_path = _optional_path(config.vis_clusters_colored_pcd_path, out_dir / "merged_vis_clusters_colored.ply")
        o3d.io.write_point_cloud(str(vis_clusters_path), vis_pcd_clusters_colored)
    if config.visualize and vis_pcd_all is not None and vis_pcd_clusters_colored is not None:
        o3d.visualization.draw_geometries([vis_pcd_all], window_name="Full scene (plant clusters colored)")
        o3d.visualization.draw_geometries([vis_pcd_clusters_colored], window_name="Plant clusters only")

    summary = {
        "input_pcd": str(input_pcd),
        "out_dir": str(out_dir),
        "points_total": n_points,
        "scene_diag": scene_diag,
        "dbscan_eps": dbscan_eps,
        "plane_dist_threshold": plane_dist_threshold,
        "plane_model": list(plane_model),
        "z_axis_inverted": bool(config.z_axis_inverted),
        "hsv_counts": {
            "plant": int(plant_mask_all.sum()),
            "leaf": int(leaf_mask_all.sum()),
            "stem": int(stem_mask_all.sum()),
        },
        "dbscan_cluster_sizes": cluster_sizes.astype(int).tolist(),
        "ground_ref": ground_ref,
        "groundlike_cluster_ids": np.where(groundlike_mask)[0].astype(int).tolist(),
        "valid_source_cluster_ids": cluster_ids.astype(int).tolist(),
        "clusters": output_clusters,
    }
    summary_path = out_dir / "summary_clustering.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d retained cluster(s) to %s", len(cluster_ids), out_dir)
    return summary


def _none_or_int(value: str | None) -> int | None:
    if value is None or str(value).lower() in {"none", "null", ""}:
        return None
    return int(value)


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the clustering CLI parser."""
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Extract plant-level clusters from a fused point cloud.")
    ap.add_argument("--input_pcd", required="input_pcd" not in defaults, default=defaults.get("input_pcd"))
    ap.add_argument("--out_dir", required="out_dir" not in defaults, default=defaults.get("out_dir"))
    ap.add_argument("--use_auto_scale", action="store_true", default=defaults.get("use_auto_scale", False))
    ap.add_argument("--z_axis_inverted", action="store_true", default=defaults.get("z_axis_inverted", False))
    ap.add_argument("--plane_dist_threshold", type=float, default=defaults.get("plane_dist_threshold", 0.005))
    ap.add_argument("--plane_dist_threshold_scale", type=float, default=defaults.get("plane_dist_threshold_scale", 0.005))
    ap.add_argument("--plane_ransac_n", type=int, default=defaults.get("plane_ransac_n", 3))
    ap.add_argument("--plane_iter", type=int, default=defaults.get("plane_iter", 1000))
    ap.add_argument("--use_nonplant_height_cut", action="store_true", default=defaults.get("use_nonplant_height_cut", False))
    ap.add_argument("--auto_height_margin_factor", type=float, default=defaults.get("auto_height_margin_factor", 4.0))
    ap.add_argument("--leaf_h_min", type=int, default=defaults.get("leaf_h_min", 30))
    ap.add_argument("--leaf_h_max", type=int, default=defaults.get("leaf_h_max", 95))
    ap.add_argument("--leaf_s_min", type=int, default=defaults.get("leaf_s_min", 60))
    ap.add_argument("--leaf_v_min", type=int, default=defaults.get("leaf_v_min", 40))
    ap.add_argument("--stem_red_1_max", type=int, default=defaults.get("stem_red_1_max", 20))
    ap.add_argument("--stem_red_2_min", type=int, default=defaults.get("stem_red_2_min", 160))
    ap.add_argument("--stem_yellow_min", type=int, default=defaults.get("stem_yellow_min", 20))
    ap.add_argument("--stem_yellow_max", type=int, default=defaults.get("stem_yellow_max", 40))
    ap.add_argument("--stem_s_min", type=int, default=defaults.get("stem_s_min", 40))
    ap.add_argument("--stem_v_min", type=int, default=defaults.get("stem_v_min", 40))
    ap.add_argument("--dbscan_eps", type=float, default=defaults.get("dbscan_eps", 0.015))
    ap.add_argument("--dbscan_eps_scale", type=float, default=defaults.get("dbscan_eps_scale", 0.02))
    ap.add_argument("--dbscan_min_pts", type=int, default=defaults.get("dbscan_min_pts", 120))
    ap.add_argument("--save_clustered_pcd", action="store_true", default=defaults.get("save_clustered_pcd", False))
    ap.add_argument("--clustered_output_path", default=defaults.get("clustered_output_path"))
    ap.add_argument("--save_each_cluster_separately", action="store_true", default=defaults.get("save_each_cluster_separately", True))
    ap.add_argument("--cluster_min_size", type=_none_or_int, default=defaults.get("cluster_min_size", 10000))
    ap.add_argument("--cluster_max_size", type=_none_or_int, default=defaults.get("cluster_max_size"))
    ap.add_argument("--save_vis_all_colored", action="store_true", default=defaults.get("save_vis_all_colored", False))
    ap.add_argument("--vis_all_colored_pcd_path", default=defaults.get("vis_all_colored_pcd_path"))
    ap.add_argument("--save_vis_clusters_colored", action="store_true", default=defaults.get("save_vis_clusters_colored", False))
    ap.add_argument("--vis_clusters_colored_pcd_path", default=defaults.get("vis_clusters_colored_pcd_path"))
    ap.add_argument("--debug_save_cluster_stats", action="store_true", default=defaults.get("debug_save_cluster_stats", False))
    ap.add_argument("--cluster_stats_path", default=defaults.get("cluster_stats_path"))
    ap.add_argument("--cluster_plot_path", default=defaults.get("cluster_plot_path"))
    ap.add_argument("--no_filter_groundlike_clusters", action="store_true", default=not defaults.get("filter_groundlike_clusters", True))
    ap.add_argument("--ground_median_margin_factor", type=float, default=defaults.get("ground_median_margin_factor", 3.0))
    ap.add_argument("--ground_height_range_factor", type=float, default=defaults.get("ground_height_range_factor", 4.0))
    ap.add_argument("--ground_planarity_ratio_max", type=float, default=defaults.get("ground_planarity_ratio_max", 0.02))
    ap.add_argument("--use_nonplant_for_plane_fit", action="store_true", default=defaults.get("use_nonplant_for_plane_fit", False))
    ap.add_argument("--visualize", action="store_true", default=defaults.get("visualize", False))
    ap.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    return ap


def config_from_args(args: argparse.Namespace) -> ClusteringConfig:
    """Convert parsed CLI args into ``ClusteringConfig``."""
    def path_or_none(value: str | None) -> Path | None:
        return Path(value) if value else None

    return ClusteringConfig(
        input_pcd=Path(args.input_pcd),
        out_dir=Path(args.out_dir),
        use_auto_scale=bool(args.use_auto_scale),
        z_axis_inverted=bool(args.z_axis_inverted),
        plane_dist_threshold=float(args.plane_dist_threshold),
        plane_dist_threshold_scale=float(args.plane_dist_threshold_scale),
        plane_ransac_n=int(args.plane_ransac_n),
        plane_iter=int(args.plane_iter),
        use_nonplant_height_cut=bool(args.use_nonplant_height_cut),
        auto_height_margin_factor=float(args.auto_height_margin_factor),
        leaf_h_min=int(args.leaf_h_min),
        leaf_h_max=int(args.leaf_h_max),
        leaf_s_min=int(args.leaf_s_min),
        leaf_v_min=int(args.leaf_v_min),
        stem_red_1_max=int(args.stem_red_1_max),
        stem_red_2_min=int(args.stem_red_2_min),
        stem_yellow_min=int(args.stem_yellow_min),
        stem_yellow_max=int(args.stem_yellow_max),
        stem_s_min=int(args.stem_s_min),
        stem_v_min=int(args.stem_v_min),
        dbscan_eps=float(args.dbscan_eps),
        dbscan_eps_scale=float(args.dbscan_eps_scale),
        dbscan_min_pts=int(args.dbscan_min_pts),
        save_clustered_pcd=bool(args.save_clustered_pcd),
        clustered_output_path=path_or_none(args.clustered_output_path),
        save_each_cluster_separately=bool(args.save_each_cluster_separately),
        cluster_min_size=args.cluster_min_size,
        cluster_max_size=args.cluster_max_size,
        save_vis_all_colored=bool(args.save_vis_all_colored),
        vis_all_colored_pcd_path=path_or_none(args.vis_all_colored_pcd_path),
        save_vis_clusters_colored=bool(args.save_vis_clusters_colored),
        vis_clusters_colored_pcd_path=path_or_none(args.vis_clusters_colored_pcd_path),
        debug_save_cluster_stats=bool(args.debug_save_cluster_stats),
        cluster_stats_path=path_or_none(args.cluster_stats_path),
        cluster_plot_path=path_or_none(args.cluster_plot_path),
        filter_groundlike_clusters=not bool(args.no_filter_groundlike_clusters),
        ground_median_margin_factor=float(args.ground_median_margin_factor),
        ground_height_range_factor=float(args.ground_height_range_factor),
        ground_planarity_ratio_max=float(args.ground_planarity_ratio_max),
        use_nonplant_for_plane_fit=bool(args.use_nonplant_for_plane_fit),
        visualize=bool(args.visualize),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``sdrawberry-cluster``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(args.verbose))
    run_clustering(config_from_args(args))


if __name__ == "__main__":
    main()

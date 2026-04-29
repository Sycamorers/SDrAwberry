"""Alpha-shape concave hull reconstruction and volume estimation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, parse_float_list, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ConcaveHullConfig:
    """Configuration for alpha-shape concave hull reconstruction."""

    in_pcd: Path
    out_dir: Path
    invert_z: bool = False
    use_scale: bool = False
    scale: float = 1.0
    remove_outlier: bool = False
    outlier_nb: int = 30
    outlier_std: float = 1.8
    alpha_voxel: float = 0.002288
    alpha_nn_sample: int = 5000
    alpha_m_list: list[float] | str = "2,3,4,5,6,7,8,9,10,15,20"
    min_triangles: int = 2000
    normalize: bool = False
    jitter_ratio: float = 0.001
    fill_holes: bool = False
    hole_size_factor: float = 3.0
    save_debug_pcd: bool = False
    keep_color: bool = False
    color_knn: int = 1
    save_only_selected: bool = False


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for concave hull reconstruction.") from exc
    return o3d


def parse_alpha_m_list(value: str | list[float] | tuple[float, ...]) -> list[float]:
    """Parse the alpha multiplier list. The public name is ``m``."""
    return parse_float_list(value)


def mesh_tag_for_m(m_value: float) -> str:
    """Return a filesystem-safe mesh tag for alpha multiplier ``m``."""
    return f"m{str(float(m_value)).rstrip('0').rstrip('.').replace('.', 'p')}"


def apply_coordinate_corrections(pcd: Any, invert_z: bool, use_scale: bool, scale: float) -> Any:
    """Apply optional Z inversion and global scale to a point cloud."""
    o3d = _require_open3d()
    points = np.asarray(pcd.points, dtype=np.float64)
    if invert_z:
        points[:, 2] *= -1.0
    if use_scale:
        points *= float(scale)
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def remove_bad_points(pcd: Any) -> Any:
    """Remove non-finite and duplicated points."""
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    return pcd


def nn_median_distance(pcd: Any, k: int = 2, sample: int = 5000, seed: int = 0) -> float:
    """Estimate median nearest-neighbor distance from a point-cloud sample."""
    o3d = _require_open3d()
    points = np.asarray(pcd.points)
    n_points = int(len(points))
    if n_points < 100:
        raise ValueError("Too few points for NN median estimate.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_points, size=sample, replace=False) if n_points > sample else np.arange(n_points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists: list[float] = []
    for index in indices:
        _, _, dist2 = tree.search_knn_vector_3d(pcd.points[int(index)], int(k))
        if len(dist2) >= 2:
            dists.append(float(np.sqrt(dist2[1])))
    if not dists:
        raise RuntimeError("Nearest-neighbor search failed to produce distances.")
    return float(np.median(np.asarray(dists)))


def normalize_pcd(pcd: Any) -> tuple[Any, np.ndarray, float]:
    """Normalize point coordinates by center and bounding-box diagonal."""
    o3d = _require_open3d()
    points = np.asarray(pcd.points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center[None, :]
    diag = float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0))) + 1e-12
    normalized = centered / diag
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized))
    return out, center, diag


def denormalize_mesh(mesh: Any, center: np.ndarray, diag: float) -> Any:
    """Move a normalized mesh back into original coordinates."""
    o3d = _require_open3d()
    out = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(out.vertices, dtype=np.float64)
    out.vertices = o3d.utility.Vector3dVector(vertices * float(diag) + center[None, :])
    return out


def clean_mesh(mesh: Any) -> Any:
    """Apply standard Open3D mesh cleanup operations."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def keep_largest_component(mesh: Any) -> Any:
    """Keep the largest connected triangle component."""
    try:
        triangle_clusters, triangle_counts, _ = mesh.cluster_connected_triangles()
        triangle_counts = np.asarray(triangle_counts)
        if triangle_counts.size == 0:
            return mesh
        largest_cluster_id = int(triangle_counts.argmax())
        remove_mask = triangle_clusters != largest_cluster_id
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    except Exception:
        return mesh
    return mesh


def mesh_stats(mesh: Any) -> tuple[int, int]:
    """Return vertex and triangle counts."""
    return int(np.asarray(mesh.vertices).shape[0]), int(np.asarray(mesh.triangles).shape[0])


def fill_holes_tensor(mesh_legacy: Any, hole_size: float) -> Any:
    """Try Open3D tensor hole filling and return the repaired legacy mesh."""
    o3d = _require_open3d()
    try:
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        return tensor_mesh.fill_holes(float(hole_size)).to_legacy()
    except Exception:
        return mesh_legacy


def transfer_mesh_vertex_colors_from_pcd(mesh: Any, pcd: Any, knn: int = 1) -> Any:
    """Transfer input point-cloud colors to mesh vertices via nearest/kNN lookup."""
    if not pcd.has_colors():
        return mesh
    o3d = _require_open3d()
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] == 0:
        return mesh

    tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    vertex_colors = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    k = max(1, int(knn))
    for i, vertex in enumerate(vertices):
        _, indices, _ = tree.search_knn_vector_3d(vertex, k)
        if len(indices) > 0:
            vertex_colors[i] = pcd_colors[np.asarray(indices)].mean(axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def try_alpha_shape(
    pcd_for_alpha: Any,
    alpha: float,
    do_fill_holes: bool = False,
    hole_size: float | None = None,
    volume_mult: float = 1.0,
    volume_unit: str = "unit^3",
) -> tuple[Any, dict[str, Any]]:
    """Build and evaluate one alpha-shape mesh."""
    o3d = _require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_for_alpha, float(alpha))
    mesh = clean_mesh(mesh)
    mesh = keep_largest_component(mesh)
    mesh = clean_mesh(mesh)

    repaired = False
    if do_fill_holes and hole_size is not None and not mesh.is_watertight():
        mesh = fill_holes_tensor(mesh, hole_size=float(hole_size))
        mesh = clean_mesh(mesh)
        mesh = keep_largest_component(mesh)
        mesh = clean_mesh(mesh)
        repaired = True

    watertight = bool(mesh.is_watertight())
    edge_manifold = bool(mesh.is_edge_manifold())
    vertices, triangles = mesh_stats(mesh)

    volume = None
    if watertight:
        try:
            volume = float(mesh.get_volume()) * float(volume_mult)
        except Exception:
            volume = None

    return mesh, {
        "method": "alpha_shape",
        "alpha": float(alpha),
        "watertight": watertight,
        "edge_manifold": edge_manifold,
        "V": vertices,
        "T": triangles,
        "volume": volume,
        "volume_unit": str(volume_unit),
        "repaired": bool(repaired),
    }


def run_concave_hull(config: ConcaveHullConfig) -> dict[str, Any]:
    """Run the alpha multiplier sweep and write ``summary_concave.json``."""
    o3d = _require_open3d()
    in_pcd = Path(config.in_pcd)
    if not in_pcd.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {in_pcd}")
    out_dir = ensure_dir(Path(config.out_dir))

    pcd0 = o3d.io.read_point_cloud(str(in_pcd))
    if pcd0.is_empty():
        raise ValueError(f"Empty point cloud: {in_pcd}")

    stem = in_pcd.stem
    pcd0 = apply_coordinate_corrections(pcd0, config.invert_z, config.use_scale, config.scale)

    if config.use_scale:
        volume_mult = 1e6
        volume_unit = "cm^3"
    else:
        volume_mult = 1.0
        volume_unit = "unit^3"

    alpha_voxel = float(config.alpha_voxel)
    if alpha_voxel <= 0:
        raise ValueError("alpha_voxel must be > 0 for stable alpha-shape reconstruction.")

    pcd_alpha = pcd0.voxel_down_sample(alpha_voxel)
    if pcd_alpha.is_empty():
        raise ValueError("Alpha-stage point cloud is empty after voxel downsampling.")

    if config.remove_outlier:
        pcd_alpha, _ = pcd_alpha.remove_statistical_outlier(
            nb_neighbors=int(config.outlier_nb),
            std_ratio=float(config.outlier_std),
        )
    pcd_alpha = remove_bad_points(pcd_alpha)
    if pcd_alpha.is_empty():
        raise ValueError("Alpha-stage point cloud is empty after cleaning.")

    nn_med = nn_median_distance(pcd_alpha, k=2, sample=int(config.alpha_nn_sample), seed=0)

    if float(config.jitter_ratio) > 0:
        points = np.asarray(pcd_alpha.points, dtype=np.float64)
        sigma = float(config.jitter_ratio) * float(nn_med)
        points = points + np.random.default_rng(0).normal(scale=sigma, size=points.shape)
        pcd_alpha.points = o3d.utility.Vector3dVector(points)

    if config.save_debug_pcd:
        o3d.io.write_point_cloud(str(out_dir / f"{stem}_alpha_pcd.ply"), pcd_alpha)

    center = None
    diag = None
    pcd_for_alpha = pcd_alpha
    if config.normalize:
        pcd_for_alpha, center, diag = normalize_pcd(pcd_alpha)
        nn_med = nn_med / float(diag)

    m_list = parse_alpha_m_list(config.alpha_m_list)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    hole_size_original = None
    hole_size_for_alpha = None
    if config.fill_holes:
        hole_size_original = float(config.hole_size_factor) * alpha_voxel
        hole_size_for_alpha = hole_size_original / float(diag) if config.normalize and diag is not None else hole_size_original

    for m_value in m_list:
        alpha = float(m_value) * nn_med
        mesh, info = try_alpha_shape(
            pcd_for_alpha,
            alpha=alpha,
            do_fill_holes=bool(config.fill_holes),
            hole_size=hole_size_for_alpha if config.fill_holes else None,
            volume_mult=volume_mult,
            volume_unit=volume_unit,
        )
        info["m"] = float(m_value)

        if config.normalize and center is not None and diag is not None:
            mesh = denormalize_mesh(mesh, center=center, diag=diag)
            mesh = clean_mesh(mesh)
            mesh = keep_largest_component(mesh)
            mesh = clean_mesh(mesh)
            info["watertight"] = bool(mesh.is_watertight())
            info["edge_manifold"] = bool(mesh.is_edge_manifold())
            info["V"], info["T"] = mesh_stats(mesh)
            if info["watertight"]:
                try:
                    info["volume"] = float(mesh.get_volume()) * float(volume_mult)
                except Exception:
                    info["volume"] = None
            else:
                info["volume"] = None
            info["volume_unit"] = str(volume_unit)

        attempts.append(dict(info))
        success = bool(info["T"] >= int(config.min_triangles) and info["watertight"] and info["volume"] is not None)
        if success:
            selected = dict(info)
            if config.keep_color:
                mesh = transfer_mesh_vertex_colors_from_pcd(mesh, pcd0, knn=int(config.color_knn))
            out_mesh = out_dir / f"{stem}_{mesh_tag_for_m(float(m_value))}.ply"
            o3d.io.write_triangle_mesh(str(out_mesh), mesh)
            selected["mesh_path"] = str(out_mesh)
            selected["mesh_has_vertex_colors"] = bool(
                len(np.asarray(mesh.vertex_colors)) == len(np.asarray(mesh.vertices))
            )
            break

        if not config.save_only_selected:
            if config.keep_color:
                mesh = transfer_mesh_vertex_colors_from_pcd(mesh, pcd0, knn=int(config.color_knn))
            out_mesh = out_dir / f"{stem}_{mesh_tag_for_m(float(m_value))}.ply"
            o3d.io.write_triangle_mesh(str(out_mesh), mesh)
            attempts[-1]["mesh_path"] = str(out_mesh)
            attempts[-1]["mesh_has_vertex_colors"] = bool(
                len(np.asarray(mesh.vertex_colors)) == len(np.asarray(mesh.vertices))
            )

    summary = {
        "input": str(in_pcd),
        "out_dir": str(out_dir),
        "coordinate_correction": {
            "invert_z": bool(config.invert_z),
            "use_scale": bool(config.use_scale),
            "scale": float(config.scale),
        },
        "volume_reporting": {
            "volume_unit": str(volume_unit),
            "volume_multiplier_applied_to_open3d_volume": float(volume_mult),
            "assumption_when_use_scale_true": "scale converts reconstruction units to meters; Open3D volume is m^3 and is converted to cm^3",
        },
        "alpha_stage": {
            "alpha_voxel": alpha_voxel,
            "remove_outlier": bool(config.remove_outlier),
            "outlier_nb": int(config.outlier_nb),
            "outlier_std": float(config.outlier_std),
            "points_alpha": int(len(pcd_alpha.points)),
            "nn_median_alpha": float(nn_med) if not config.normalize else float(nn_med * float(diag)),
            "normalize_used": bool(config.normalize),
            "jitter_ratio": float(config.jitter_ratio),
        },
        "repair": {
            "fill_holes": bool(config.fill_holes),
            "hole_size_factor": float(config.hole_size_factor),
            "hole_size_original_coordinates": float(hole_size_original) if hole_size_original is not None else None,
            "hole_size_alpha_coordinates": float(hole_size_for_alpha) if hole_size_for_alpha is not None else None,
        },
        "color_transfer": {
            "keep_color": bool(config.keep_color),
            "color_knn": int(config.color_knn),
            "input_has_colors": bool(pcd0.has_colors()),
            "note": "colors are assigned to mesh vertex_colors via nearest/kNN from the input point cloud",
        },
        "output_control": {
            "save_only_selected": bool(config.save_only_selected),
            "mesh_naming": "<input_pcd_name>_m*.ply",
            "legacy_cli_alias": "--alpha_k_list is accepted as an alias for --alpha_m_list",
        },
        "alpha_sweep": {
            "m_list": m_list,
            "attempts": attempts,
            "selected": selected,
        },
    }

    out_json = out_dir / "summary_concave.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if selected is None:
        LOGGER.warning("No watertight concave hull mesh found. See %s", out_json)
    else:
        LOGGER.info("Selected m=%s volume=%s %s", selected["m"], selected["volume"], selected["volume_unit"])
    return summary


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the concave hull CLI parser."""
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Reconstruct alpha-shape concave hull meshes and estimate volume.")
    ap.add_argument("--in_pcd", required="in_pcd" not in defaults, default=defaults.get("in_pcd"))
    ap.add_argument("--out_dir", required="out_dir" not in defaults, default=defaults.get("out_dir"))
    ap.add_argument("--invert_z", action="store_true", default=defaults.get("invert_z", False))
    ap.add_argument("--use_scale", action="store_true", default=defaults.get("use_scale", False))
    ap.add_argument("--scale", type=float, default=defaults.get("scale", 1.0))
    ap.add_argument("--remove_outlier", action="store_true", default=defaults.get("remove_outlier", False))
    ap.add_argument("--outlier_nb", type=int, default=defaults.get("outlier_nb", 30))
    ap.add_argument("--outlier_std", type=float, default=defaults.get("outlier_std", 1.8))
    ap.add_argument("--alpha_voxel", type=float, default=defaults.get("alpha_voxel", 0.002288))
    ap.add_argument("--alpha_nn_sample", type=int, default=defaults.get("alpha_nn_sample", 5000))
    ap.add_argument(
        "--alpha_m_list",
        "--alpha_k_list",
        dest="alpha_m_list",
        default=defaults.get("alpha_m_list", "2,3,4,5,6,7,8,9,10,15,20"),
        help="Comma-separated alpha multipliers m. The legacy --alpha_k_list alias is accepted.",
    )
    ap.add_argument("--min_triangles", type=int, default=defaults.get("min_triangles", 2000))
    ap.add_argument("--normalize", action="store_true", default=defaults.get("normalize", False))
    ap.add_argument("--jitter_ratio", type=float, default=defaults.get("jitter_ratio", 0.001))
    ap.add_argument("--fill_holes", action="store_true", default=defaults.get("fill_holes", False))
    ap.add_argument("--hole_size_factor", type=float, default=defaults.get("hole_size_factor", 3.0))
    ap.add_argument("--save_debug_pcd", action="store_true", default=defaults.get("save_debug_pcd", False))
    ap.add_argument("--keep_color", action="store_true", default=defaults.get("keep_color", False))
    ap.add_argument("--color_knn", type=int, default=defaults.get("color_knn", 1))
    ap.add_argument("--save_only_selected", action="store_true", default=defaults.get("save_only_selected", False))
    ap.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    return ap


def config_from_args(args: argparse.Namespace) -> ConcaveHullConfig:
    """Convert parsed args to ``ConcaveHullConfig``."""
    return ConcaveHullConfig(
        in_pcd=Path(args.in_pcd),
        out_dir=Path(args.out_dir),
        invert_z=bool(args.invert_z),
        use_scale=bool(args.use_scale),
        scale=float(args.scale),
        remove_outlier=bool(args.remove_outlier),
        outlier_nb=int(args.outlier_nb),
        outlier_std=float(args.outlier_std),
        alpha_voxel=float(args.alpha_voxel),
        alpha_nn_sample=int(args.alpha_nn_sample),
        alpha_m_list=args.alpha_m_list,
        min_triangles=int(args.min_triangles),
        normalize=bool(args.normalize),
        jitter_ratio=float(args.jitter_ratio),
        fill_holes=bool(args.fill_holes),
        hole_size_factor=float(args.hole_size_factor),
        save_debug_pcd=bool(args.save_debug_pcd),
        keep_color=bool(args.keep_color),
        color_knn=int(args.color_knn),
        save_only_selected=bool(args.save_only_selected),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``sdrawberry-concave-hull``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(args.verbose))
    run_concave_hull(config_from_args(args))


if __name__ == "__main__":
    main()

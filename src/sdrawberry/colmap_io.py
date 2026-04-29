"""COLMAP model loading utilities.

The package intentionally does not vendor COLMAP's ``read_write_model.py`` helper.
For binary/text sparse models, install ``pycolmap`` and use these wrappers.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def model_files_exist(model_dir: Path) -> bool:
    """Return True when a COLMAP sparse model appears to exist in *model_dir*."""
    names = {p.name for p in model_dir.iterdir()} if model_dir.is_dir() else set()
    has_bin = {"cameras.bin", "images.bin", "points3D.bin"}.issubset(names)
    has_txt = {"cameras.txt", "images.txt", "points3D.txt"}.issubset(names)
    return has_bin or has_txt


def _iter_mapping_values(obj: Any) -> list[Any]:
    if hasattr(obj, "values"):
        return list(obj.values())
    return list(obj)


def _image_id(image: Any, fallback: int | None = None) -> int:
    for name in ("image_id", "id"):
        if hasattr(image, name):
            return int(getattr(image, name))
    if fallback is None:
        raise AttributeError("Could not determine COLMAP image id.")
    return int(fallback)


def _point_id(point: Any, fallback: int | None = None) -> int:
    for name in ("point3D_id", "id"):
        if hasattr(point, name):
            return int(getattr(point, name))
    if fallback is None:
        raise AttributeError("Could not determine COLMAP point3D id.")
    return int(fallback)


def _image_xys(image: Any) -> list[tuple[float, float]]:
    if hasattr(image, "xys"):
        arr = np.asarray(image.xys, dtype=np.float64)
        return [tuple(round(float(v), 6) for v in xy) for xy in arr]

    if hasattr(image, "points2D"):
        xys: list[tuple[float, float]] = []
        for point2d in image.points2D:
            xy = np.asarray(point2d.xy, dtype=np.float64)
            xys.append(tuple(round(float(v), 6) for v in xy))
        return xys

    raise AttributeError(f"Could not read 2D observations for image {getattr(image, 'name', '<unknown>')}")


def _point_track_elements(point: Any) -> list[tuple[int, int]]:
    if hasattr(point, "image_ids") and hasattr(point, "point2D_idxs"):
        return [(int(i), int(j)) for i, j in zip(point.image_ids, point.point2D_idxs)]

    track = getattr(point, "track", None)
    elements = getattr(track, "elements", None)
    if elements is None:
        return []

    out: list[tuple[int, int]] = []
    for elem in elements:
        image_id = getattr(elem, "image_id", None)
        point2d_idx = getattr(elem, "point2D_idx", None)
        if point2d_idx is None:
            point2d_idx = getattr(elem, "point2D_id", None)
        if image_id is not None and point2d_idx is not None:
            out.append((int(image_id), int(point2d_idx)))
    return out


def load_views_and_points_from_colmap(model_dir: Path, ext: str = "") -> tuple[dict[int, np.ndarray], dict[tuple[str, tuple[float, float]], list[int]]]:
    """Load a COLMAP sparse model as 3D points and image-feature view indices.

    Returns:
        points: mapping from COLMAP point3D id to XYZ coordinates.
        view_index: mapping from ``(image_name, rounded_xy_2d)`` to point3D ids.

    The ``ext`` argument is accepted for compatibility with older scripts. PyCOLMAP
    auto-detects binary/text models from the model directory.
    """
    del ext
    model_dir = Path(model_dir)
    if not model_files_exist(model_dir):
        raise FileNotFoundError(
            "Missing COLMAP sparse model. Expected cameras/images/points3D as .bin or .txt in "
            f"{model_dir}"
        )

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "pycolmap is required to read COLMAP sparse models. Install it or provide preprocessed "
            "correspondences before running fusion."
        ) from exc

    reconstruction = pycolmap.Reconstruction(str(model_dir))

    image_id_to_name: dict[int, str] = {}
    image_points: dict[str, list[tuple[float, float]]] = {}

    images = getattr(reconstruction, "images", {})
    if hasattr(images, "items"):
        image_items = list(images.items())
    else:
        image_items = [(None, img) for img in images]

    for fallback_id, image in image_items:
        image_id = _image_id(image, fallback=fallback_id)
        image_id_to_name[image_id] = str(image.name)
        image_points[str(image.name)] = _image_xys(image)

    points: dict[int, np.ndarray] = {}
    view_index: dict[tuple[str, tuple[float, float]], list[int]] = defaultdict(list)

    points3d = getattr(reconstruction, "points3D", {})
    if hasattr(points3d, "items"):
        point_items = list(points3d.items())
    else:
        point_items = [(None, point) for point in points3d]

    for fallback_id, point in point_items:
        point_id = _point_id(point, fallback=fallback_id)
        xyz = np.asarray(point.xyz, dtype=np.float64)
        points[point_id] = xyz

        for image_id, point2d_idx in _point_track_elements(point):
            image_name = image_id_to_name.get(int(image_id))
            if image_name is None:
                continue
            xys = image_points.get(image_name, [])
            if 0 <= int(point2d_idx) < len(xys):
                view_index[(image_name, xys[int(point2d_idx)])].append(point_id)

    return points, dict(view_index)


def export_points_ply_from_colmap_model(model_dir: Path, ply_path: Path) -> None:
    """Export COLMAP ``points3D`` from *model_dir* to a colored PLY point cloud."""
    model_dir = Path(model_dir)
    ply_path = Path(ply_path)
    if not model_files_exist(model_dir):
        raise FileNotFoundError(
            "Cannot export PLY because the COLMAP sparse model is missing in "
            f"{model_dir}"
        )
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap is required for COLMAP-to-PLY export.") from exc

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    points3d = getattr(reconstruction, "points3D", {})
    for point in _iter_mapping_values(points3d):
        pts.append(np.asarray(point.xyz, dtype=np.float32))
        color = getattr(point, "color", np.array([255, 255, 255], dtype=np.uint8))
        cols.append(np.asarray(color, dtype=np.uint8))

    if not pts:
        raise ValueError(f"No points3D found in COLMAP model: {model_dir}")

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    pts_arr = np.asarray(pts, dtype=np.float32)
    cols_arr = np.asarray(cols, dtype=np.uint8)

    try:
        import trimesh

        trimesh.PointCloud(vertices=pts_arr, colors=cols_arr).export(str(ply_path))
        return
    except Exception:
        pass

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Install trimesh or open3d to write exported PLY files.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_arr.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_arr.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)

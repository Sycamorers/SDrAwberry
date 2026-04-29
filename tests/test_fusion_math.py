import math

import numpy as np
import pytest

from sdrawberry.fusion import (
    VoxelHash,
    estimate_similarity_transform,
    find_matches_from_views,
    roll_pairs_refined,
)


def test_estimate_similarity_transform_unit_scale() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.5, -0.2, 1.0])
    target = source + translation

    scale, rotation, estimated_translation = estimate_similarity_transform(source, target)
    assert scale == pytest.approx(1.0)
    assert np.allclose(rotation, np.eye(3), atol=1e-8)
    assert np.allclose(estimated_translation, translation, atol=1e-8)


def test_roll_pairs_refined_recovers_y_axis_rotation() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    theta = 0.25
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    translation = np.array([0.2, -0.3, 0.4])
    target = (rotation @ source.T).T + translation

    transform = roll_pairs_refined(source, target, max_iter=5)
    transformed = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    assert np.allclose(transformed, target, atol=1e-8)


def test_find_matches_from_views() -> None:
    view = ("0001_cam_a.png", (10.0, 20.0))
    views_a = {view: [1]}
    views_b = {view: [2]}
    points_a = {1: np.array([0.0, 0.0, 0.0])}
    points_b = {2: np.array([1.0, 0.0, 0.0])}

    x_a, x_b, matched_b = find_matches_from_views(views_a, views_b, points_a, points_b, k=50)
    assert x_a.shape == (1, 3)
    assert x_b.shape == (1, 3)
    assert matched_b.shape == (1, 3)


def test_voxel_hash_accumulates_centroids_without_open3d() -> None:
    voxels = VoxelHash(1.0)
    voxels.add_points(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.2, 0.0, 0.0]]))
    assert len(voxels.map) == 2
    first = voxels.map[(0, 0, 0)]
    assert first[-1] == 2
    assert first[0] / first[-1] == pytest.approx(0.15)

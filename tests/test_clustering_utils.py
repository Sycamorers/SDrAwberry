import numpy as np
import pytest

from sdrawberry.clustering import cluster_planarity_ratio, is_groundlike_cluster, normalize_plane, signed_plane_distances


def test_normalize_plane_and_signed_distances() -> None:
    plane = normalize_plane((0.0, 0.0, 2.0, -4.0))
    assert plane == pytest.approx((0.0, 0.0, 1.0, -2.0))

    points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
    distances, flipped_plane = signed_plane_distances(points, plane, z_axis_inverted=True)
    assert distances.tolist() == pytest.approx([-1.0, 1.0])
    assert flipped_plane == pytest.approx((-0.0, -0.0, -1.0, 2.0))


def test_cluster_planarity_ratio_for_flat_points() -> None:
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    assert cluster_planarity_ratio(points) < 1e-10


def test_groundlike_cluster_rule() -> None:
    points = np.column_stack(
        [
            np.linspace(0, 1, 20),
            np.linspace(0, 0.2, 20),
            np.zeros(20),
        ]
    )
    distances = np.full(20, 0.001)
    assert is_groundlike_cluster(
        points=points,
        distances=distances,
        ground_ref=0.0,
        plane_dist_threshold=0.005,
        median_margin_factor=3.0,
        height_range_factor=4.0,
        planarity_ratio_max=0.02,
    )

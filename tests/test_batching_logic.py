from pathlib import Path

import pytest

from sdrawberry.batch import (
    _parse_frame_id_from_name,
    batch_folder_name,
    build_batches,
    detect_cameras,
    group_by_frame,
    parse_batch_range,
)


def test_parse_frame_id_preserves_leading_zeros() -> None:
    assert _parse_frame_id_from_name("rgb_00042.png") == (42, "00042")
    with pytest.raises(ValueError):
        _parse_frame_id_from_name("frame_42.png")


def test_group_and_build_overlapping_batches(tmp_path: Path) -> None:
    for cam in ["cam_b", "cam_a"]:
        rgb_dir = tmp_path / "images" / cam / "rgb"
        rgb_dir.mkdir(parents=True)
        for frame in ["0001", "0002", "0003", "0004"]:
            (rgb_dir / f"rgb_{frame}.png").write_bytes(b"")

    images_dir = tmp_path / "images"
    assert detect_cameras(images_dir) == ["cam_a", "cam_b"]

    grouped = group_by_frame(images_dir)
    assert list(grouped) == [1, 2, 3, 4]
    assert [p.parent.parent.name for p in grouped[1]] == ["cam_a", "cam_b"]

    batches = build_batches(list(grouped), grouped, batch_size=3, overlap=1)
    assert len(batches) == 2
    assert len(batches[0]) == 6
    assert len(batches[1]) == 4


def test_batch_range_and_folder_name() -> None:
    assert parse_batch_range("3-8") == (3, 8)
    assert parse_batch_range("8:9") == (8, 9)
    assert batch_folder_name(1) == "01"
    assert batch_folder_name(100) == "100"
    with pytest.raises(ValueError):
        batch_folder_name(0)

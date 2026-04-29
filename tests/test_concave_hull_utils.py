import pytest

from sdrawberry.concave_hull import mesh_tag_for_m, parse_alpha_m_list


def test_parse_alpha_m_list() -> None:
    assert parse_alpha_m_list("2,3.5, 10") == [2.0, 3.5, 10.0]
    assert parse_alpha_m_list([1, 2]) == [1.0, 2.0]
    with pytest.raises(ValueError):
        parse_alpha_m_list("")


def test_mesh_tag_for_m() -> None:
    assert mesh_tag_for_m(2.0) == "m2"
    assert mesh_tag_for_m(3.5) == "m3p5"

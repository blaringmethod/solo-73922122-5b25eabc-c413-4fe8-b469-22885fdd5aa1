"""Functional tests: initial state, replacement, pyramid aggregation,
replacement semantics, validation and restart persistence."""


from app import db
from tests.conftest import make_tile, partial_tile


def test_initial_grid_is_all_null(conn):
    cells = db.query_window(conn, 0, 0, 0, 1024, 1024)
    assert len(cells) == 1024 * 1024
    assert all(c["valid_count"] == 0 and c["mean"] is None for c in cells)

    for level in range(1, db.LEVELS):
        dim = 1024 >> level
        cells = db.query_window(conn, level, 0, 0, dim, dim)
        assert all(c["valid_count"] == 0 and c["mean"] is None for c in cells)


def test_replace_tile_level0_and_level1(conn):
    db.replace_tile(conn, 0, 0, make_tile(50), "2026-01-01T00:00:00+00:00")

    # raw pixels
    cells = db.query_window(conn, 0, 0, 0, 16, 16)
    assert len(cells) == 256
    assert all(c["valid_count"] == 1 and c["mean"] == 50.0 for c in cells)

    # level 1: 8x8 coarse cells inside the tile, each holds 4 valid pixels
    cells = db.query_window(conn, 1, 0, 0, 8, 8)
    assert len(cells) == 64
    assert all(c["valid_count"] == 4 and c["mean"] == 50.0 for c in cells)

    # neighbouring level-1 cells stay empty
    cells = db.query_window(conn, 1, 8, 0, 9, 1)
    assert cells[0] == {"x": 8, "y": 0, "valid_count": 0, "mean": None}


def test_full_pyramid_after_replacing_all_tiles(conn):
    for ty in range(64):
        for tx in range(64):
            db.replace_tile(conn, tx, ty, make_tile(80), "t")

    # every coarse level must be fully covered with correct counts/means
    for level in range(1, db.LEVELS):
        dim = 1024 >> level
        cells = db.query_window(conn, level, 0, 0, dim, dim)
        footprint = (1 << level) ** 2  # original pixels per coarse cell
        assert len(cells) == dim * dim
        assert all(c["valid_count"] == footprint for c in cells)
        assert all(c["mean"] == 80.0 for c in cells)

    top = db.query_window(conn, 10, 0, 0, 1, 1)[0]
    assert top == {"x": 0, "y": 0, "valid_count": 1_048_576, "mean": 80.0}


def test_null_pixels_excluded_from_count_and_mean(conn):
    # half the pixels are 10, half are null
    db.replace_tile(conn, 2, 3, partial_tile(10), "t")

    cells = db.query_window(conn, 1, 16, 24, 24, 32)
    assert len(cells) == 64
    # each 2x2 block of a checkerboard has exactly 2 valid pixels
    assert all(c["valid_count"] == 2 for c in cells)
    assert all(c["mean"] == 10.0 for c in cells)

    # tile (2,3) base pixel = (32,48); level-5 cell covering it is (1,1)
    l5 = db.query_window(conn, 5, 1, 1, 2, 2)[0]
    # 256 pixels in the tile, half valid -> 128
    assert l5["valid_count"] == 128
    assert l5["mean"] == 10.0


def test_mean_is_average_of_mixed_values(conn):
    values = [[(x * 7 + y * 3) % 101 for x in range(16)] for y in range(16)]
    db.replace_tile(conn, 0, 0, values, "t")

    flat = [v for row in values for v in row]
    expected_mean = round(sum(flat) / len(flat), 4)

    top = db.query_window(conn, 5, 0, 0, 1, 1)[0]
    assert top["valid_count"] == 256
    assert top["mean"] == expected_mean


def test_replacement_overwrites_and_recomputes(conn):
    db.replace_tile(conn, 0, 0, make_tile(100), "t1")
    db.replace_tile(conn, 0, 0, make_tile(20), "t2")

    raw = db.query_window(conn, 0, 0, 0, 16, 16)
    assert all(c["mean"] == 20.0 for c in raw)

    l1 = db.query_window(conn, 1, 0, 0, 8, 8)
    assert all(c["valid_count"] == 4 and c["mean"] == 20.0 for c in l1)

    tile = db.get_tile(conn, 0, 0)
    assert tile["updated_at"] == "t2"
    assert tile["values"][0][0] == 20


def test_replacing_with_nulls_empties_summaries(conn):
    db.replace_tile(conn, 0, 0, make_tile(90), "t1")
    db.replace_tile(conn, 0, 0, make_tile(None), "t2")

    l1 = db.query_window(conn, 1, 0, 0, 8, 8)
    assert all(c["valid_count"] == 0 and c["mean"] is None for c in l1)
    l5 = db.query_window(conn, 5, 0, 0, 1, 1)[0]
    assert l5["valid_count"] == 0 and l5["mean"] is None
    # tile row still exists (an explicit all-null correction)
    assert db.get_tile(conn, 0, 0)["values"][0][0] is None


def test_other_tiles_are_not_touched_in_pyramid(conn):
    db.replace_tile(conn, 0, 0, make_tile(40), "t1")
    db.replace_tile(conn, 1, 0, make_tile(60), "t2")

    # level-5 cell (0,0) aggregates both 16x16 tiles: 512 valid pixels
    l5 = db.query_window(conn, 5, 0, 0, 1, 1)[0]
    assert l5["valid_count"] == 512
    assert l5["mean"] == 50.0

    # level-1 cells still reflect each tile's own value separately
    left = db.query_window(conn, 1, 0, 0, 8, 8)
    right = db.query_window(conn, 1, 8, 0, 16, 8)
    assert all(c["mean"] == 40.0 for c in left)
    assert all(c["mean"] == 60.0 for c in right)


def test_invalid_inputs_rejected(conn):
    import pytest

    with pytest.raises(ValueError):
        db.replace_tile(conn, 64, 0, make_tile(1), "t")
    with pytest.raises(ValueError):
        db.replace_tile(conn, 0, -1, make_tile(1), "t")
    with pytest.raises(ValueError):
        db.replace_tile(conn, 0, 0, [[1] * 15] * 16, "t")

    with pytest.raises(ValueError):
        db.query_window(conn, 11, 0, 0, 1, 1)
    with pytest.raises(ValueError):
        db.query_window(conn, 0, 0, 0, 1025, 1)


def test_persistence_after_reopen(db_path):
    conn = db.connect(db_path)
    db.init_db(db_path)
    db.replace_tile(conn, 5, 7, make_tile(33), "persisted")
    conn.close()

    # reopen: simulates a service restart against the same database file
    conn2 = db.connect(db_path)
    try:
        raw = db.query_window(conn2, 0, 80, 112, 96, 128)
        assert all(c["mean"] == 33.0 and c["valid_count"] == 1 for c in raw)
        top = db.query_window(conn2, 10, 0, 0, 1, 1)[0]
        assert top["valid_count"] == 256
        assert db.get_tile(conn2, 5, 7)["updated_at"] == "persisted"
    finally:
        conn2.close()

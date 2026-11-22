"""Two-period (baseline vs current) comparison tests.

Covers exact pairing counts, uncovered tiles as missing data, summary-only
change queries, incremental refresh scope, concurrent corrections of both
periods at the same position, rollback, and restart persistence.
"""

import multiprocessing as mp
import sqlite3
import threading
import time

import pytest

from app import db
from tests.conftest import make_tile, partial_tile


# --------------------------------------------------------------------------- #
# pairing and aggregation
# --------------------------------------------------------------------------- #


def test_unsubmitted_period_counts_as_missing(conn):
    db.replace_tile(conn, 2, 3, make_tile(60), "c")
    # baseline was never submitted: every current pixel is "current only"
    cell = db.query_change_window(conn, 5, 1, 1, 2, 2)[0]
    assert cell == {
        "x": 1, "y": 1,
        "both_count": 0,
        "baseline_only_count": 0,
        "current_only_count": 256,
        "mean_change_pp": None,
    }


def test_baseline_only_and_both_counts(conn):
    db.replace_tile(conn, 2, 3, make_tile(70), "c")
    db.replace_tile(conn, 2, 3, make_tile(40), "b", db.BASELINE_PERIOD)

    cell = db.query_change_window(conn, 5, 1, 1, 2, 2)[0]
    assert cell["both_count"] == 256
    assert cell["baseline_only_count"] == 0
    assert cell["current_only_count"] == 0
    assert cell["mean_change_pp"] == 30.0

    # a baseline correction at a position with no current tile is baseline-only
    db.replace_tile(conn, 4, 3, make_tile(40), "b", db.BASELINE_PERIOD)
    cell = db.query_change_window(conn, 5, 2, 1, 3, 2)[0]
    assert cell["both_count"] == 0
    assert cell["baseline_only_count"] == 256
    assert cell["current_only_count"] == 0
    assert cell["mean_change_pp"] is None


def test_partial_null_pairing_at_level1(conn):
    # checkerboard: 128 valid pixels per tile, on the same coordinates in both
    # periods, so all 128 are paired and the difference is 5 percentage points
    db.replace_tile(conn, 2, 3, partial_tile(15), "c")
    db.replace_tile(conn, 2, 3, partial_tile(10), "b", db.BASELINE_PERIOD)

    cells = db.query_change_window(conn, 1, 16, 24, 24, 32)
    assert len(cells) == 64
    for cell in cells:
        assert cell["both_count"] == 2
        assert cell["baseline_only_count"] == 0
        assert cell["current_only_count"] == 0
        assert cell["mean_change_pp"] == 5.0


def test_disjoint_validity_counts_and_null_mean(conn):
    # current valid where baseline is null and vice versa (bitwise opposite)
    current = [
        [20 if (x + y) % 2 == 0 else None for x in range(16)]
        for y in range(16)
    ]
    baseline = [
        [None if (x + y) % 2 == 0 else 70 for x in range(16)]
        for y in range(16)
    ]
    db.replace_tile(conn, 0, 0, current, "c")
    db.replace_tile(conn, 0, 0, baseline, "b", db.BASELINE_PERIOD)

    cells = db.query_change_window(conn, 1, 0, 0, 8, 8)
    assert all(c["both_count"] == 0 for c in cells)
    assert all(c["current_only_count"] == 2 for c in cells)
    assert all(c["baseline_only_count"] == 2 for c in cells)
    assert all(c["mean_change_pp"] is None for c in cells)


def test_coarse_levels_aggregate_change_counts(conn):
    # two tiles under the same level-5 cell, corrected in both periods
    db.replace_tile(conn, 0, 0, make_tile(60), "c")
    db.replace_tile(conn, 1, 0, make_tile(30), "c")
    db.replace_tile(conn, 0, 0, make_tile(50), "b", db.BASELINE_PERIOD)
    # tile (1,0) has no baseline tile -> only current pixels

    cell = db.query_change_window(conn, 5, 0, 0, 1, 1)[0]
    assert cell["both_count"] == 256          # tile (0,0) paired
    assert cell["current_only_count"] == 256  # tile (1,0) current only
    assert cell["baseline_only_count"] == 0
    assert cell["mean_change_pp"] == 10.0     # (60 - 50) over paired pixels

    top = db.query_change_window(conn, 10, 0, 0, 1, 1)[0]
    assert top["both_count"] == 256
    assert top["current_only_count"] == 256
    assert top["mean_change_pp"] == 10.0


def test_change_requires_level_1_to_10(conn):
    with pytest.raises(ValueError):
        db.query_change_window(conn, 0, 0, 0, 1024, 1024)
    with pytest.raises(ValueError):
        db.query_change_window(conn, 11, 0, 0, 1, 1)
    with pytest.raises(ValueError):
        db.query_change_window(conn, 5, 0, 0, 33, 1)


def test_counts_match_ground_truth_at_every_level(conn):
    """Coarse cells equal the exact category counts over their raw footprints."""
    import random

    rng = random.Random(1234)
    current, baseline = {}, {}
    for ty, tx in [(0, 0), (0, 1), (1, 0), (3, 3), (7, 6)]:
        cur = [[None if rng.random() < 0.3 else rng.randint(0, 100)
                for _ in range(16)] for _ in range(16)]
        base = [[None if rng.random() < 0.4 else rng.randint(0, 100)
                 for _ in range(16)] for _ in range(16)]
        current[(tx, ty)] = cur
        baseline[(tx, ty)] = base
        db.replace_tile(conn, tx, ty, cur, "c")
        db.replace_tile(conn, tx, ty, base, "b", db.BASELINE_PERIOD)

    for level in range(1, db.LEVELS):
        span = 1 << level
        dim = db.GRID >> level
        cells = db.query_change_window(conn, level, 0, 0, dim, dim)
        assert len(cells) == dim * dim
        for cell in cells:
            both = base_only = cur_only = diff = 0
            for py in range(cell["y"] * span, (cell["y"] + 1) * span):
                for px in range(cell["x"] * span, (cell["x"] + 1) * span):
                    tx, ty = px // 16, py // 16
                    cur = current.get((tx, ty))
                    base = baseline.get((tx, ty))
                    cv = None if cur is None else cur[py % 16][px % 16]
                    bv = None if base is None else base[py % 16][px % 16]
                    if cv is not None and bv is not None:
                        both += 1
                        diff += cv - bv
                    elif cv is not None:
                        cur_only += 1
                    elif bv is not None:
                        base_only += 1
            assert cell["both_count"] == both, (level, cell)
            assert cell["baseline_only_count"] == base_only, (level, cell)
            assert cell["current_only_count"] == cur_only, (level, cell)
            if both:
                assert cell["mean_change_pp"] == round(diff / both, 4)
            else:
                assert cell["mean_change_pp"] is None


def test_periods_are_isolated(conn):
    db.replace_tile(conn, 0, 0, make_tile(11), "c")
    db.replace_tile(conn, 0, 0, make_tile(88), "b", db.BASELINE_PERIOD)

    assert db.query_window(conn, 1, 0, 0, 8, 8)[0]["mean"] == 11.0
    assert db.query_window(
        conn, 1, 0, 0, 8, 8, db.BASELINE_PERIOD
    )[0]["mean"] == 88.0
    assert db.get_tile(conn, 0, 0)["values"][0][0] == 11
    assert db.get_tile(
        conn, 0, 0, db.BASELINE_PERIOD
    )["values"][0][0] == 88


def test_correction_to_missing_clears_change_cell(conn):
    db.replace_tile(conn, 0, 0, make_tile(40), "c")
    db.replace_tile(conn, 0, 0, make_tile(30), "b", db.BASELINE_PERIOD)
    assert db.query_change_window(conn, 5, 0, 0, 1, 1)[0]["both_count"] == 256

    # correcting the baseline back to all-null must not leave stale pairings
    db.replace_tile(conn, 0, 0, make_tile(None), "b2", db.BASELINE_PERIOD)
    cell = db.query_change_window(conn, 5, 0, 0, 1, 1)[0]
    assert cell["both_count"] == 0
    assert cell["current_only_count"] == 256
    assert cell["baseline_only_count"] == 0
    assert cell["mean_change_pp"] is None

    # both periods explicitly null: every comparison cell is empty
    db.replace_tile(conn, 0, 0, make_tile(None), "c2")
    for level in (1, 5, 10):
        cell = db.query_change_window(conn, level, 0, 0, 1, 1)[0]
        assert cell["both_count"] == 0
        assert cell["baseline_only_count"] == 0
        assert cell["current_only_count"] == 0
        assert cell["mean_change_pp"] is None


# --------------------------------------------------------------------------- #
# read/write scope
# --------------------------------------------------------------------------- #


def test_change_query_reads_only_summaries(conn):
    db.replace_tile(conn, 0, 0, make_tile(10), "c")
    db.replace_tile(conn, 0, 0, make_tile(20), "b", db.BASELINE_PERIOD)

    tables: set[str] = set()

    def tracer(statement):
        s = " ".join(statement.upper().split())
        for table in ("TILES", "SUMMARIES", "CHANGE_SUMMARIES"):
            if f"FROM {table}" in s:
                tables.add(table)
        return None

    conn.set_trace_callback(tracer)
    db.query_change_window(conn, 5, 0, 0, 1, 1)
    conn.set_trace_callback(None)

    assert tables == {"CHANGE_SUMMARIES"}


def _overlaps(level: int, cx: int, cy: int, tile_x: int, tile_y: int) -> bool:
    span = 1 << level
    return (
        cx * span < (tile_x + 1) * db.TILE_PX
        and (cx + 1) * span > tile_x * db.TILE_PX
        and cy * span < (tile_y + 1) * db.TILE_PX
        and (cy + 1) * span > tile_y * db.TILE_PX
    )


def _change_rows(conn):
    rows = conn.execute(
        """
        SELECT level, cx, cy, both_cnt, baseline_only_cnt, current_only_cnt,
               diff_sum
        FROM change_summaries
        """
    ).fetchall()
    return {(r["level"], r["cx"], r["cy"]): tuple(r)[3:] for r in rows}


def test_correction_refreshes_only_affected_summaries(conn):
    db.replace_tile(conn, 0, 0, make_tile(30), "c")
    db.replace_tile(conn, 0, 0, make_tile(20), "b", db.BASELINE_PERIOD)
    db.replace_tile(conn, 20, 20, make_tile(80), "c")

    before = _change_rows(conn)
    db.replace_tile(conn, 0, 0, make_tile(45), "b2", db.BASELINE_PERIOD)
    after = _change_rows(conn)

    changed = {
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    assert changed  # the correction did update the comparison
    assert all(
        _overlaps(level, cx, cy, 0, 0) for level, cx, cy in changed
    )
    # nothing at the unrelated tile's level-4 cell changed
    assert before[(4, 20, 20)] == after[(4, 20, 20)]
    assert before[(5, 10, 10)] == after[(5, 10, 10)]


def test_correction_reads_only_same_position_raw_tile(conn):
    db.replace_tile(conn, 0, 0, make_tile(30), "c")
    db.replace_tile(conn, 2, 0, make_tile(30), "c")
    db.replace_tile(conn, 0, 0, make_tile(20), "b", db.BASELINE_PERIOD)

    selects: list[str] = []

    def tracer(statement):
        s = " ".join(statement.upper().split())
        if " FROM TILES" in s:
            selects.append(s)
        return None

    conn.set_trace_callback(tracer)
    db.replace_tile(conn, 0, 0, make_tile(25), "b2", db.BASELINE_PERIOD)
    conn.set_trace_callback(None)

    assert selects  # the counterpart period at the same position was read
    assert all("BETWEEN" not in s for s in selects)
    assert all("TILE_X = 0" in s and "TILE_Y = 0" in s for s in selects)


# --------------------------------------------------------------------------- #
# concurrency, snapshot, rollback, persistence
# --------------------------------------------------------------------------- #


def _worker(db_path, period, value, delay):
    conn = db.connect(db_path)
    try:
        time.sleep(delay)
        db.replace_tile(conn, 0, 0, make_tile(value), f"{period}", period)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "first,second",
    [(db.BASELINE_PERIOD, db.CURRENT_PERIOD),
     (db.CURRENT_PERIOD, db.BASELINE_PERIOD)],
)
def test_two_processes_correct_both_periods_same_position(
    tmp_path, first, second
):
    path = str(tmp_path / f"two-{first}-{second}.db")
    db.init_db(path)
    values = {db.BASELINE_PERIOD: 25, db.CURRENT_PERIOD: 65}

    p1 = mp.Process(target=_worker, args=(path, first, values[first], 0.0))
    p2 = mp.Process(target=_worker, args=(path, second, values[second], 0.05))
    p1.start()
    p2.start()
    p1.join(30)
    p2.join(30)
    assert p1.exitcode == 0 and p2.exitcode == 0

    conn = db.connect(path)
    try:
        cell = db.query_change_window(conn, 5, 0, 0, 1, 1)[0]
        # both commits count: 256 paired pixels, current - baseline = 40pp
        assert cell["both_count"] == 256
        assert cell["baseline_only_count"] == 0
        assert cell["current_only_count"] == 0
        assert cell["mean_change_pp"] == 40.0
    finally:
        conn.close()


def test_change_reader_never_sees_partial_state(tmp_path):
    path = str(tmp_path / "change-snapshot.db")
    db.init_db(path)
    conn = db.connect(path)
    db.replace_tile(conn, 0, 0, make_tile(40), "b", db.BASELINE_PERIOD)
    conn.close()

    stop = threading.Event()

    def writer():
        wconn = db.connect(path)
        i = 0
        try:
            while not stop.is_set():
                value = None if i % 2 else (10 if (i // 2) % 2 else 90)
                db.replace_tile(wconn, 0, 0, make_tile(value), "c")
                i += 1
        finally:
            wconn.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        rconn = db.connect(path)
        deadline = time.time() + 4
        while time.time() < deadline:
            cell = db.query_change_window(rconn, 5, 0, 0, 1, 1)[0]
            if cell["both_count"] == 256:
                assert cell["current_only_count"] == 0
                assert cell["baseline_only_count"] == 0
                assert cell["mean_change_pp"] in (-30.0, 50.0)
            else:
                # a committed all-null current tile: baseline is unpaired
                assert cell["both_count"] == 0
                assert cell["current_only_count"] == 0
                assert cell["baseline_only_count"] == 256
                assert cell["mean_change_pp"] is None
        rconn.close()
    finally:
        stop.set()
        thread.join(30)


def test_failed_correction_keeps_old_tile_and_change(tmp_path):
    path = str(tmp_path / "change-rollback.db")
    db.init_db(path)
    conn = db.connect(path)
    db.replace_tile(conn, 0, 0, make_tile(30), "c")
    db.replace_tile(conn, 0, 0, make_tile(20), "b", db.BASELINE_PERIOD)

    real_execute = conn.execute
    calls = {"n": 0}

    def flaky_execute(sql, params=()):
        if isinstance(sql, str) and "INSERT INTO change_summaries" in sql:
            calls["n"] += 1
            if calls["n"] == 5:  # fail partway through the refresh
                raise sqlite3.OperationalError("simulated failure")
        return real_execute(sql, params)

    class FailingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            return flaky_execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    with pytest.raises(sqlite3.OperationalError):
        db.replace_tile(
            FailingConn(conn), 0, 0, make_tile(99), "b2", db.BASELINE_PERIOD
        )

    # old baseline tile, current tile and both pyramids survive
    assert db.get_tile(conn, 0, 0, db.BASELINE_PERIOD)["values"][0][0] == 20
    assert db.get_tile(conn, 0, 0)["values"][0][0] == 30
    cell = db.query_change_window(conn, 5, 0, 0, 1, 1)[0]
    assert cell["both_count"] == 256 and cell["mean_change_pp"] == 10.0
    conn.close()

    # restart: committed data (not the failed correction) is served
    conn2 = db.connect(path)
    try:
        assert db.get_tile(
            conn2, 0, 0, db.BASELINE_PERIOD
        )["updated_at"] == "b"
        assert db.query_change_window(
            conn2, 5, 0, 0, 1, 1
        )[0]["mean_change_pp"] == 10.0
    finally:
        conn2.close()


def test_baseline_and_change_survive_reopen(db_path):
    db.init_db(db_path)
    conn = db.connect(db_path)
    db.replace_tile(conn, 5, 7, make_tile(70), "c")
    db.replace_tile(conn, 5, 7, make_tile(55), "b", db.BASELINE_PERIOD)
    conn.close()

    conn2 = db.connect(db_path)
    try:
        cell = db.query_change_window(conn2, 5, 2, 3, 3, 4)[0]
        assert cell["both_count"] == 256
        assert cell["mean_change_pp"] == 15.0
        assert db.query_window(
            conn2, 5, 2, 3, 3, 4, db.BASELINE_PERIOD
        )[0]["mean"] == 55.0
    finally:
        conn2.close()


# --------------------------------------------------------------------------- #
# migration of an existing single-period database
# --------------------------------------------------------------------------- #

LEGACY_SCHEMA = """
CREATE TABLE tiles (
    tile_t INTEGER NOT NULL, tile_x INTEGER NOT NULL, tile_y INTEGER NOT NULL,
    data BLOB NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (tile_t, tile_x, tile_y)
) WITHOUT ROWID;
CREATE TABLE summaries (
    level INTEGER NOT NULL, cx INTEGER NOT NULL, cy INTEGER NOT NULL,
    cnt INTEGER NOT NULL, total INTEGER NOT NULL,
    PRIMARY KEY (level, cx, cy)
) WITHOUT ROWID;
"""


def test_existing_single_period_db_becomes_current(db_path):
    """Data written by the original service is the current period; the change
    summaries are backfilled once, with the baseline counted as missing."""
    legacy = db.connect(db_path)
    legacy.executescript(LEGACY_SCHEMA)
    legacy.execute(
        "INSERT INTO tiles VALUES (0, 5, 7, ?, 'old')",
        (db.encode_tile(make_tile(64)),),
    )
    for by in range(0, 16, 2):
        for bx in range(0, 16, 2):
            legacy.execute(
                "INSERT INTO summaries VALUES (1, ?, ?, 4, 256)",
                (80 // 2 + bx // 2, 112 // 2 + by // 2),
            )
    for level in range(2, db.LEVELS):
        # single level-1 cell per level: enough to route the original query
        legacy.execute(
            "INSERT INTO summaries VALUES (?, 2, 3, 256, 16384)",
            (level,),
        )
    legacy.close()

    db.init_db(db_path)  # adds the two-period tables and backfills
    conn = db.connect(db_path)
    try:
        # the original current-period data is still served as before
        assert db.query_window(conn, 5, 2, 3, 3, 4)[0]["mean"] == 64.0
        cell = db.query_change_window(conn, 5, 2, 3, 3, 4)[0]
        assert cell["both_count"] == 0
        assert cell["current_only_count"] == 256
        assert cell["baseline_only_count"] == 0
        assert cell["mean_change_pp"] is None
    finally:
        conn.close()

    # a second startup does not duplicate or reset the backfill
    db.init_db(db_path)
    conn2 = db.connect(db_path)
    try:
        assert db.query_change_window(
            conn2, 10, 0, 0, 1, 1
        )[0]["current_only_count"] == 256
    finally:
        conn2.close()


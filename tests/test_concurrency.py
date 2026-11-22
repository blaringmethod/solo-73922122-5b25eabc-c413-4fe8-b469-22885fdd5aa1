"""Concurrency tests: two independent connections/processes replacing
different tiles under the same coarse cell must both be counted, and readers
must only ever observe complete committed states.
"""

import multiprocessing as mp
import sqlite3
import time

from app import db
from tests.conftest import make_tile


def _worker(db_path: str, tile_x: int, value: int, delay: float) -> None:
    """Runs in a fresh process: a second SQLite instance on the same file."""
    conn = db.connect(db_path)
    try:
        # stagger starts so the two BEGIN IMMEDIATE transactions overlap
        time.sleep(delay)
        db.replace_tile(conn, tile_x, 0, make_tile(value), "t")
    finally:
        conn.close()


def test_two_processes_replace_siblings_both_counted(tmp_path):
    path = str(tmp_path / "multi.db")
    db.init_db(path)

    # tiles (0,0) and (1,0) share level-1..5 ancestor cells
    p1 = mp.Process(target=_worker, args=(path, 0, 20, 0.0))
    p2 = mp.Process(target=_worker, args=(path, 1, 60, 0.05))
    p1.start()
    p2.start()
    p1.join(30)
    p2.join(30)
    assert p1.exitcode == 0 and p2.exitcode == 0

    conn = db.connect(path)
    try:
        l5 = db.query_window(conn, 5, 0, 0, 1, 1)[0]
        assert l5["valid_count"] == 512  # both 16x16 tiles, no lost update
        assert l5["mean"] == 40.0        # (256*20 + 256*60) / 512

        # level-1 cells keep each tile's own values
        assert all(
            c["mean"] == 20.0
            for c in db.query_window(conn, 1, 0, 0, 8, 8)
        )
        assert all(
            c["mean"] == 60.0
            for c in db.query_window(conn, 1, 8, 0, 16, 8)
        )
    finally:
        conn.close()


def test_reader_never_sees_partial_state(tmp_path):
    path = str(tmp_path / "reader.db")
    db.init_db(path)

    stop = mp.Event()

    def writer():
        conn = db.connect(path)
        i = 0
        try:
            while not stop.is_set():
                # alternate values on the same tile; every committed state is
                # either the old full tile or the new full tile, never a mix
                db.replace_tile(conn, 0, 0, make_tile(i % 100), "t")
                i += 1
        finally:
            conn.close()

    proc = mp.Process(target=writer)
    proc.start()
    try:
        conn = db.connect(path)
        deadline = time.time() + 5
        while time.time() < deadline:
            cells = db.query_window(conn, 1, 0, 0, 8, 8)
            counts = {c["valid_count"] for c in cells}
            means = {c["mean"] for c in cells}
            # snapshot consistency: all 64 level-1 cells share one state
            if means == {None}:
                assert counts == {0}
            else:
                assert None not in means
                assert counts == {4}
                assert len(means) == 1  # uniform tile -> one mean in snapshot
        conn.close()
    finally:
        stop.set()
        proc.join(30)
        assert proc.exitcode == 0


def test_rollback_keeps_previous_tile_and_summaries(tmp_path):
    path = str(tmp_path / "rollback.db")
    db.init_db(path)
    conn = db.connect(path)

    db.replace_tile(conn, 0, 0, make_tile(70), "v1")

    real_execute = conn.execute
    calls = {"n": 0}

    def flaky_execute(sql, params=()):
        # fail inside the second transaction, while writing a level-3 summary
        if isinstance(sql, str) and "INSERT INTO summaries" in sql:
            calls["n"] += 1
            if calls["n"] == (64 + 16 + 1):  # during the first level-3 upsert
                raise sqlite3.OperationalError("simulated writer failure")
        return real_execute(sql, params)

    # sqlite3.Connection.execute is read-only, so wrap the connection in an
    # object that delegates everything except execute, which we sabotage.
    class FailingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            return flaky_execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    import pytest

    with pytest.raises(sqlite3.OperationalError):
        db.replace_tile(FailingConn(conn), 0, 0, make_tile(10), "v2")

    # old tile and its summaries survive intact
    raw = db.query_window(conn, 0, 0, 0, 16, 16)
    assert all(c["mean"] == 70.0 for c in raw)
    l1 = db.query_window(conn, 1, 0, 0, 8, 8)
    assert all(c["mean"] == 70.0 and c["valid_count"] == 4 for c in l1)
    l5 = db.query_window(conn, 5, 0, 0, 1, 1)[0]
    assert l5["mean"] == 70.0 and l5["valid_count"] == 256
    assert db.get_tile(conn, 0, 0)["updated_at"] == "v1"
    conn.close()


def test_replacement_does_not_scan_other_tiles_raw_pixels(tmp_path):
    """Statement trace proves refresh of tile (1,0) reads no other tile BLOB."""
    path = str(tmp_path / "trace.db")
    db.init_db(path)
    conn = db.connect(path)
    db.replace_tile(conn, 0, 0, make_tile(50), "t0")
    db.replace_tile(conn, 2, 0, make_tile(50), "t2")

    scanned_tables: list[str] = set()

    def tracer(statement):
        s = statement.upper()
        if " FROM TILES" in s:
            scanned_tables.add("tiles")
        if " FROM SUMMARIES" in s:
            scanned_tables.add("summaries")
        return None

    conn.set_trace_callback(tracer)
    db.replace_tile(conn, 1, 0, make_tile(90), "t1")
    conn.set_trace_callback(None)

    # the replacement upserts its own tile but never SELECTs the tiles table:
    # higher levels are rebuilt from summaries only
    assert "tiles" not in scanned_tables
    assert "summaries" in scanned_tables

    # tiles (0,0) and (1,0) sit in level-5 cell (0,0); tile (2,0) in (1,0).
    # A level-4 cell covers exactly one 16x16 tile.  All three tiles span 48
    # raw pixels, so they first share the 64x64 level-6 cell (0,0).
    l4 = db.query_window(conn, 4, 0, 0, 1, 1)[0]
    assert l4["valid_count"] == 256
    l6 = db.query_window(conn, 6, 0, 0, 1, 1)[0]
    assert l6["valid_count"] == 3 * 256
    conn.close()

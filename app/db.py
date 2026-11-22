"""SQLite storage for the 1024x1024 equal-area canopy-cover raster pyramid.

Layout
------
The 1024x1024 level-0 grid is partitioned into 64x64 tiles of 16x16 pixels.
A replacement covers one whole tile at a time.

Coarse levels 1..10 are formed by merging every adjacent 2x2 block of cells of
the level below into one cell (1024 -> 512 -> ... -> 1).  Each coarse cell
stores the number of *valid original pixels* in its footprint and the sum of
their coverage values, so its mean is sum/count (or null when count = 0).

Concurrency / durability
------------------------
Every replacement runs in a single ``BEGIN IMMEDIATE`` transaction, so two
processes (e.g. two API containers sharing one database file) replacing
different tiles under the same coarse cell serialise on the write lock and
both corrections are accumulated.  Any failure rolls the whole transaction
back, leaving the previous tile and summaries intact.  WAL mode gives readers a
consistent snapshot while a writer is in progress.
"""

from __future__ import annotations

import array
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Sequence

GRID = 1024
TILE_PX = 16
TILES_PER_AXIS = GRID // TILE_PX  # 64
LEVELS = 11  # level 0 (1024) .. level 10 (1)
NIL = 0x7FFF  # int16 sentinel stored in the tile BLOB for null pixels

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_t INTEGER NOT NULL,
    tile_x INTEGER NOT NULL,
    tile_y INTEGER NOT NULL,
    data   BLOB NOT NULL,            -- 256 signed 16-bit values, row-major
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tile_t, tile_x, tile_y)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS summaries (
    level INTEGER NOT NULL,          -- 1..10
    cx    INTEGER NOT NULL,
    cy    INTEGER NOT NULL,
    cnt   INTEGER NOT NULL,          -- valid original pixels in the footprint
    total INTEGER NOT NULL,          -- sum of their coverage values
    PRIMARY KEY (level, cx, cy)
) WITHOUT ROWID;
"""

# --------------------------------------------------------------------------- #
# connection management
# --------------------------------------------------------------------------- #


def connect(db_path: str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a connection configured for a shared multi-process database.

    ``check_same_thread=False`` is used by the web pool together with a
    per-connection lock so that a connection may safely outlive the worker
    thread that first opened it (thread pools recycle OS threads).
    """
    directory = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=15.0,
        isolation_level=None,
        check_same_thread=check_same_thread,
    )
    conn.row_factory = sqlite3.Row
    # WAL lets readers and one writer work concurrently; busy_timeout makes a
    # second writer wait for the first one's IMMEDIATE lock instead of failing.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE: take the reserved lock up front, always finalise."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --------------------------------------------------------------------------- #
# tile (de)serialisation
# --------------------------------------------------------------------------- #


def encode_tile(values: Sequence[Sequence[int | None]]) -> bytes:
    """Encode a 16x16 matrix (int 0..100 or None) as a signed int16 BLOB."""
    flat: list[int] = []
    for row in values:
        for v in row:
            flat.append(NIL if v is None else int(v))
    return array.array("h", flat).tobytes()


def decode_tile(blob: bytes) -> list[list[int | None]]:
    raw = array.array("h")
    raw.frombytes(blob)
    grid = [[None] * TILE_PX for _ in range(TILE_PX)]
    for idx, v in enumerate(raw):
        if v != NIL:
            grid[idx // TILE_PX][idx % TILE_PX] = int(v)
    return grid


# --------------------------------------------------------------------------- #
# write path: whole-tile replacement + incremental pyramid refresh
# --------------------------------------------------------------------------- #


def _upsert_summary(
    conn: sqlite3.Connection, level: int, cx: int, cy: int, cnt: int, total: int
) -> None:
    conn.execute(
        """
        INSERT INTO summaries(level, cx, cy, cnt, total)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(level, cx, cy) DO UPDATE SET cnt = excluded.cnt,
                                                 total = excluded.total
        """,
        (level, cx, cy, cnt, total),
    )


def replace_tile(
    conn: sqlite3.Connection,
    tile_x: int,
    tile_y: int,
    values: Sequence[Sequence[int | None]],
    now_iso: str,
) -> None:
    """Replace one 16x16 tile and refresh only the pyramid cells it touches.

    Level-1 summaries are recomputed from the new raw pixels; every higher
    level is recomputed purely from the four child summaries, so raw pixels of
    other tiles are never scanned.  The whole operation is one atomic
    transaction: on failure nothing changes.
    """
    if not (0 <= tile_x < TILES_PER_AXIS and 0 <= tile_y < TILES_PER_AXIS):
        raise ValueError(
            f"tile coordinates must be within 0..{TILES_PER_AXIS - 1}"
        )
    if len(values) != TILE_PX or any(len(r) != TILE_PX for r in values):
        raise ValueError(f"tile payload must be {TILE_PX}x{TILE_PX}")

    blob = encode_tile(values)
    base_x = tile_x * TILE_PX
    base_y = tile_y * TILE_PX

    with _transaction(conn):
        # 1) whole-tile replacement (upsert keeps it atomic with the summaries)
        conn.execute(
            """
            INSERT INTO tiles(tile_t, tile_x, tile_y, data, updated_at)
            VALUES (0, ?, ?, ?, ?)
            ON CONFLICT(tile_t, tile_x, tile_y)
            DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
            """,
            (tile_x, tile_y, blob, now_iso),
        )

        # 2) level 1: recompute the 8x8 coarse cells covered by this tile
        #    straight from the new raw pixels.
        for by in range(0, TILE_PX, 2):
            for bx in range(0, TILE_PX, 2):
                cnt = 0
                total = 0
                for dy in range(2):
                    for dx in range(2):
                        v = values[by + dy][bx + dx]
                        if v is not None:
                            cnt += 1
                            total += int(v)
                _upsert_summary(
                    conn,
                    1,
                    (base_x + bx) // 2,
                    (base_y + by) // 2,
                    cnt,
                    total,
                )

        # 3) levels 2..10: each touched parent is rebuilt by merging ALL FOUR
        #    of its child summary rows (including children owned by other
        #    tiles).  Only ancestors of this tile are recomputed; no raw
        #    pixels outside the replaced tile are ever read.
        for level in range(2, LEVELS):
            # parent cells on this level whose footprint overlaps the tile
            px0 = base_x >> level
            py0 = base_y >> level
            px1 = (base_x + TILE_PX - 1) >> level
            py1 = (base_y + TILE_PX - 1) >> level
            for py in range(py0, py1 + 1):
                for px in range(px0, px1 + 1):
                    child_x0, child_y0 = px * 2, py * 2
                    row = conn.execute(
                        """
                        SELECT COALESCE(SUM(cnt), 0)   AS cnt,
                               COALESCE(SUM(total), 0) AS total
                        FROM summaries
                        WHERE level = ?
                          AND cx IN (?, ?)
                          AND cy IN (?, ?)
                        """,
                        (level - 1, child_x0, child_x0 + 1,
                         child_y0, child_y0 + 1),
                    ).fetchone()
                    _upsert_summary(
                        conn, level, px, py, row["cnt"], row["total"]
                    )


# --------------------------------------------------------------------------- #
# read path
# --------------------------------------------------------------------------- #


def _level0_window(
    conn: sqlite3.Connection, x0: int, y0: int, x1: int, y1: int
) -> list[dict]:
    """Read raw pixels; only the tiles intersecting the window are touched."""
    tx0, tx1 = x0 // TILE_PX, (x1 - 1) // TILE_PX
    ty0, ty1 = y0 // TILE_PX, (y1 - 1) // TILE_PX
    tiles: dict[tuple[int, int], list[list[int | None]]] = {}
    rows = conn.execute(
        """
        SELECT tile_x, tile_y, data FROM tiles
        WHERE tile_t = 0 AND tile_x BETWEEN ? AND ?
                       AND tile_y BETWEEN ? AND ?
        """,
        (tx0, tx1, ty0, ty1),
    ).fetchall()
    for r in rows:
        tiles[(r["tile_x"], r["tile_y"])] = decode_tile(r["data"])

    out: list[dict] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            tile = tiles.get((x // TILE_PX, y // TILE_PX))
            v = None
            if tile is not None:
                v = tile[y % TILE_PX][x % TILE_PX]
            if v is None:
                out.append({"x": x, "y": y, "valid_count": 0, "mean": None})
            else:
                # a level-0 cell is one original pixel: count 1, mean == value
                out.append(
                    {"x": x, "y": y, "valid_count": 1, "mean": float(v)}
                )
    return out


def query_window(
    conn: sqlite3.Connection,
    level: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[dict]:
    """Return every cell of ``level`` inside [x0,x1) x [y0,y1).

    Each row carries the valid original-pixel count and the mean coverage;
    the mean is null when no valid original pixel falls in the footprint.
    The read runs in one deferred transaction so a single query always sees
    one committed snapshot, even while another process is committing.
    """
    if not (0 <= level < LEVELS):
        raise ValueError(f"level must be within 0..{LEVELS - 1}")
    dim = GRID >> level
    if not (0 <= x0 < x1 <= dim and 0 <= y0 < y1 <= dim):
        raise ValueError(f"window must be within the {dim}x{dim} level grid")

    conn.execute("BEGIN")
    try:
        if level == 0:
            return _level0_window(conn, x0, y0, x1, y1)
        rows = conn.execute(
            """
            SELECT cx, cy, cnt, total FROM summaries
            WHERE level = ? AND cx BETWEEN ? AND ?
                          AND cy BETWEEN ? AND ?
            """,
            (level, x0, x1 - 1, y0, y1 - 1),
        ).fetchall()
        stored = {(r["cx"], r["cy"]): (r["cnt"], r["total"]) for r in rows}
        out: list[dict] = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                cnt, total = stored.get((x, y), (0, 0))
                mean = round(total / cnt, 4) if cnt else None
                out.append(
                    {"x": x, "y": y, "valid_count": cnt, "mean": mean}
                )
        return out
    finally:
        # SELECT-only transaction; COMMIT releases the snapshot.
        conn.execute("COMMIT")


def get_tile(
    conn: sqlite3.Connection, tile_x: int, tile_y: int
) -> dict | None:
    """Fetch one tile as the submitted 16x16 matrix, or None if never written."""
    if not (0 <= tile_x < TILES_PER_AXIS and 0 <= tile_y < TILES_PER_AXIS):
        raise ValueError(
            f"tile coordinates must be within 0..{TILES_PER_AXIS - 1}"
        )
    row = conn.execute(
        """
        SELECT data, updated_at FROM tiles
        WHERE tile_t = 0 AND tile_x = ? AND tile_y = ?
        """,
        (tile_x, tile_y),
    ).fetchone()
    if row is None:
        return None
    return {
        "tile_x": tile_x,
        "tile_y": tile_y,
        "values": decode_tile(row["data"]),
        "updated_at": row["updated_at"],
    }

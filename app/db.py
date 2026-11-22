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

Two periods
-----------
``tile_t = 0`` holds the current period (the original data) and ``tile_t = 1``
an optional baseline period, using identical coordinates and payloads.  Each
period has its own pyramid table, plus ``change_summaries`` which pairs the
original pixels of the two periods cell by cell.  Replacing one tile refreshes
its own period's pyramid and the comparison cells at that exact position only;
summaries never need to read raw pixels of neighbouring tiles.
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

# The existing dataset is the current period.  A baseline period is stored in
# the same ``tiles`` table under a different ``tile_t`` value, so both periods
# use exactly the same tile coordinates and payload format.
CURRENT_PERIOD = 0
BASELINE_PERIOD = 1
PERIODS = (CURRENT_PERIOD, BASELINE_PERIOD)
PERIOD_NAMES = {CURRENT_PERIOD: "current", BASELINE_PERIOD: "baseline"}

# Only internal, fixed names are interpolated into SQL.
_SUMMARY_TABLE = {
    CURRENT_PERIOD: "summaries",
    BASELINE_PERIOD: "baseline_summaries",
}
_CHANGE_BACKFILL_KEY = "change_summaries_backfilled_v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_t INTEGER NOT NULL,         -- 0 = current period, 1 = baseline
    tile_x INTEGER NOT NULL,
    tile_y INTEGER NOT NULL,
    data   BLOB NOT NULL,            -- 256 signed 16-bit values, row-major
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tile_t, tile_x, tile_y)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS summaries (
    level INTEGER NOT NULL,          -- 1..10, current period
    cx    INTEGER NOT NULL,
    cy    INTEGER NOT NULL,
    cnt   INTEGER NOT NULL,          -- valid original pixels in the footprint
    total INTEGER NOT NULL,          -- sum of their coverage values
    PRIMARY KEY (level, cx, cy)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS baseline_summaries (
    level INTEGER NOT NULL,          -- 1..10, baseline period
    cx    INTEGER NOT NULL,
    cy    INTEGER NOT NULL,
    cnt   INTEGER NOT NULL,
    total INTEGER NOT NULL,
    PRIMARY KEY (level, cx, cy)
) WITHOUT ROWID;

-- Per-cell comparison of the two periods, built from original pixels and then
-- rolled up like the ordinary pyramid.  Counts are exact original-pixel
-- counts; diff_sum is the sum of (current - baseline) over paired pixels.
CREATE TABLE IF NOT EXISTS change_summaries (
    level             INTEGER NOT NULL,  -- 1..10
    cx                INTEGER NOT NULL,
    cy                INTEGER NOT NULL,
    both_cnt          INTEGER NOT NULL,  -- valid in both periods
    baseline_only_cnt INTEGER NOT NULL,  -- valid only in baseline
    current_only_cnt  INTEGER NOT NULL,  -- valid only in current
    diff_sum          INTEGER NOT NULL,  -- sum(current - baseline), paired
    PRIMARY KEY (level, cx, cy)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
        # Databases created by the original single-period service already hold
        # current-period data.  Backfill the comparison summaries once so
        # existing corrections immediately participate in change queries.
        # BEGIN IMMEDIATE serialises two instances starting at the same time.
        with _transaction(conn):
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = ?",
                (_CHANGE_BACKFILL_KEY,),
            ).fetchone()
            if row is None:
                _backfill_change_summaries(conn)
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                    (_CHANGE_BACKFILL_KEY, "1"),
                )


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


def _summary_table(period: int) -> str:
    try:
        return _SUMMARY_TABLE[period]
    except KeyError:
        raise ValueError("period must be current (0) or baseline (1)") from None


def _upsert_summary(
    conn: sqlite3.Connection, level: int, cx: int, cy: int, cnt: int, total: int,
    period: int = CURRENT_PERIOD,
) -> None:
    table = _summary_table(period)
    conn.execute(
        f"""
        INSERT INTO {table}(level, cx, cy, cnt, total)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(level, cx, cy) DO UPDATE SET cnt = excluded.cnt,
                                                 total = excluded.total
        """,
        (level, cx, cy, cnt, total),
    )


def _upsert_change(
    conn: sqlite3.Connection,
    level: int,
    cx: int,
    cy: int,
    both: int,
    baseline_only: int,
    current_only: int,
    diff_sum: int,
) -> None:
    if not (both or baseline_only or current_only):
        # Empty footprints carry no comparison; keep the summary sparse and
        # remove a row that may have held data before this correction.
        conn.execute(
            "DELETE FROM change_summaries WHERE level = ? AND cx = ? AND cy = ?",
            (level, cx, cy),
        )
        return
    conn.execute(
        """
        INSERT INTO change_summaries(
            level, cx, cy, both_cnt, baseline_only_cnt, current_only_cnt,
            diff_sum
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(level, cx, cy) DO UPDATE SET
            both_cnt = excluded.both_cnt,
            baseline_only_cnt = excluded.baseline_only_cnt,
            current_only_cnt = excluded.current_only_cnt,
            diff_sum = excluded.diff_sum
        """,
        (level, cx, cy, both, baseline_only, current_only, diff_sum),
    )


def _get_tile_values(
    conn: sqlite3.Connection, period: int, tile_x: int, tile_y: int
) -> list[list[int | None]] | None:
    """Read one period's raw tile at exactly this position (no other tile)."""
    row = conn.execute(
        """
        SELECT data FROM tiles
        WHERE tile_t = ? AND tile_x = ? AND tile_y = ?
        """,
        (period, tile_x, tile_y),
    ).fetchone()
    return None if row is None else decode_tile(row["data"])


def _refresh_period_pyramid(
    conn: sqlite3.Connection,
    period: int,
    tile_x: int,
    tile_y: int,
    values: Sequence[Sequence[int | None]],
) -> None:
    """Rebuild level 1 of one period from ``values`` and levels 2..10 from
    child summaries.  Nothing outside this tile's ancestors is read."""
    base_x = tile_x * TILE_PX
    base_y = tile_y * TILE_PX

    # level 1: the 8x8 coarse cells covered by this tile, from raw pixels
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
                period,
            )

    # levels 2..10: each touched parent merges all four child summary rows,
    # including children owned by other tiles; no raw pixel is read here.
    for level in range(2, LEVELS):
        px0 = base_x >> level
        py0 = base_y >> level
        px1 = (base_x + TILE_PX - 1) >> level
        py1 = (base_y + TILE_PX - 1) >> level
        for py in range(py0, py1 + 1):
            for px in range(px0, px1 + 1):
                child_x0, child_y0 = px * 2, py * 2
                row = conn.execute(
                    f"""
                    SELECT COALESCE(SUM(cnt), 0)   AS cnt,
                           COALESCE(SUM(total), 0) AS total
                    FROM {_summary_table(period)}
                    WHERE level = ?
                      AND cx IN (?, ?)
                      AND cy IN (?, ?)
                    """,
                    (level - 1, child_x0, child_x0 + 1,
                     child_y0, child_y0 + 1),
                ).fetchone()
                _upsert_summary(
                    conn, level, px, py, row["cnt"], row["total"], period
                )


def _refresh_change_pyramid(
    conn: sqlite3.Connection,
    tile_x: int,
    tile_y: int,
    replaced_period: int,
    replaced_values: Sequence[Sequence[int | None]],
) -> None:
    """Rebuild the comparison cells touched by one tile position.

    Only the two raw tiles at *this* position are considered: the replacement
    payload plus the counterpart period's tile at the same coordinates.  No
    other tile's raw pixels are read; levels 2..10 are rebuilt from child
    summary rows only.
    """
    base_x = tile_x * TILE_PX
    base_y = tile_y * TILE_PX
    other_period = (
        BASELINE_PERIOD if replaced_period == CURRENT_PERIOD
        else CURRENT_PERIOD
    )
    other = _get_tile_values(conn, other_period, tile_x, tile_y)
    current: Sequence[Sequence[int | None]] | None
    baseline: Sequence[Sequence[int | None]] | None
    if replaced_period == CURRENT_PERIOD:
        current, baseline = replaced_values, other
    else:
        current, baseline = other, replaced_values

    def at(tile: Sequence[Sequence[int | None]] | None, y: int, x: int):
        return None if tile is None else tile[y][x]

    # level 1: exact per-original-pixel pairing inside this tile
    for by in range(0, TILE_PX, 2):
        for bx in range(0, TILE_PX, 2):
            both = 0
            baseline_only = 0
            current_only = 0
            diff_sum = 0
            for dy in range(2):
                for dx in range(2):
                    cur = at(current, by + dy, bx + dx)
                    base = at(baseline, by + dy, bx + dx)
                    if cur is not None and base is not None:
                        both += 1
                        diff_sum += int(cur) - int(base)
                    elif cur is not None:
                        current_only += 1
                    elif base is not None:
                        baseline_only += 1
            _upsert_change(
                conn,
                1,
                (base_x + bx) // 2,
                (base_y + by) // 2,
                both,
                baseline_only,
                current_only,
                diff_sum,
            )

    # levels 2..10: sum the four child comparison rows
    for level in range(2, LEVELS):
        px0 = base_x >> level
        py0 = base_y >> level
        px1 = (base_x + TILE_PX - 1) >> level
        py1 = (base_y + TILE_PX - 1) >> level
        for py in range(py0, py1 + 1):
            for px in range(px0, px1 + 1):
                child_x0, child_y0 = px * 2, py * 2
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(both_cnt), 0)          AS both_cnt,
                           COALESCE(SUM(baseline_only_cnt), 0)  AS baseline_only_cnt,
                           COALESCE(SUM(current_only_cnt), 0)   AS current_only_cnt,
                           COALESCE(SUM(diff_sum), 0)           AS diff_sum
                    FROM change_summaries
                    WHERE level = ?
                      AND cx IN (?, ?)
                      AND cy IN (?, ?)
                    """,
                    (level - 1, child_x0, child_x0 + 1,
                     child_y0, child_y0 + 1),
                ).fetchone()
                _upsert_change(
                    conn,
                    level,
                    px,
                    py,
                    row["both_cnt"],
                    row["baseline_only_cnt"],
                    row["current_only_cnt"],
                    row["diff_sum"],
                )


def replace_tile(
    conn: sqlite3.Connection,
    tile_x: int,
    tile_y: int,
    values: Sequence[Sequence[int | None]],
    now_iso: str,
    period: int = CURRENT_PERIOD,
) -> None:
    """Replace one 16x16 tile of one period and refresh affected summaries.

    A period's pyramid level 1 is recomputed from the submitted raw pixels;
    higher levels are recomputed purely from the four child summaries.  The
    comparison pyramid is refreshed in the same transaction by reading only
    the two raw tiles (current and baseline) at this exact position.  The
    whole operation is atomic: on failure nothing changes.
    """
    if period not in PERIODS:
        raise ValueError("period must be current (0) or baseline (1)")
    if not (0 <= tile_x < TILES_PER_AXIS and 0 <= tile_y < TILES_PER_AXIS):
        raise ValueError(
            f"tile coordinates must be within 0..{TILES_PER_AXIS - 1}"
        )
    if len(values) != TILE_PX or any(len(r) != TILE_PX for r in values):
        raise ValueError(f"tile payload must be {TILE_PX}x{TILE_PX}")

    blob = encode_tile(values)

    with _transaction(conn):
        # 1) whole-tile replacement (upsert keeps it atomic with the summaries)
        conn.execute(
            """
            INSERT INTO tiles(tile_t, tile_x, tile_y, data, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tile_t, tile_x, tile_y)
            DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
            """,
            (period, tile_x, tile_y, blob, now_iso),
        )

        # 2) this period's pyramid
        _refresh_period_pyramid(conn, period, tile_x, tile_y, values)

        # 3) the two-period comparison pyramid at this same position
        _refresh_change_pyramid(
            conn, tile_x, tile_y, period, values
        )


# --------------------------------------------------------------------------- #
# one-time migration for databases created by the single-period service
# --------------------------------------------------------------------------- #


def _backfill_change_summaries(conn: sqlite3.Connection) -> None:
    """Build ``change_summaries`` from already-committed raw tiles, once.

    Older databases hold current-period data but no comparison rows; this
    makes existing corrections immediately visible to the change API.  New
    databases simply have nothing to backfill.
    """
    conn.execute("DELETE FROM change_summaries")
    positions = conn.execute(
        """
        SELECT DISTINCT tile_x, tile_y FROM tiles
        WHERE tile_t IN (?, ?)
        """,
        (CURRENT_PERIOD, BASELINE_PERIOD),
    ).fetchall()
    if not positions:
        return

    # level 1 comes from the raw tiles at each populated position only
    for pos in positions:
        tx, ty = pos["tile_x"], pos["tile_y"]
        current = _get_tile_values(conn, CURRENT_PERIOD, tx, ty)
        baseline = _get_tile_values(conn, BASELINE_PERIOD, tx, ty)

        def at(tile: list[list[int | None]] | None, y: int, x: int):
            return None if tile is None else tile[y][x]

        for by in range(0, TILE_PX, 2):
            for bx in range(0, TILE_PX, 2):
                both = baseline_only = current_only = diff_sum = 0
                for dy in range(2):
                    for dx in range(2):
                        cur = at(current, by + dy, bx + dx)
                        base = at(baseline, by + dy, bx + dx)
                        if cur is not None and base is not None:
                            both += 1
                            diff_sum += int(cur) - int(base)
                        elif cur is not None:
                            current_only += 1
                        elif base is not None:
                            baseline_only += 1
                if both or baseline_only or current_only:
                    _upsert_change(
                        conn,
                        1,
                        (tx * TILE_PX + bx) // 2,
                        (ty * TILE_PX + by) // 2,
                        both,
                        baseline_only,
                        current_only,
                        diff_sum,
                    )

    # levels 2..10: only parents of cells that carry comparison data
    for level in range(2, LEVELS):
        parents = conn.execute(
            f"""
            SELECT DISTINCT cx / 2 AS px, cy / 2 AS py
            FROM change_summaries
            WHERE level = ?
            """,
            (level - 1,),
        ).fetchall()
        for parent in parents:
            px, py = parent["px"], parent["py"]
            child_x0, child_y0 = px * 2, py * 2
            row = conn.execute(
                """
                SELECT COALESCE(SUM(both_cnt), 0)          AS both_cnt,
                       COALESCE(SUM(baseline_only_cnt), 0)  AS baseline_only_cnt,
                       COALESCE(SUM(current_only_cnt), 0)   AS current_only_cnt,
                       COALESCE(SUM(diff_sum), 0)           AS diff_sum
                FROM change_summaries
                WHERE level = ?
                  AND cx IN (?, ?)
                  AND cy IN (?, ?)
                """,
                (level - 1, child_x0, child_x0 + 1,
                 child_y0, child_y0 + 1),
            ).fetchone()
            _upsert_change(
                conn, level, px, py,
                row["both_cnt"], row["baseline_only_cnt"],
                row["current_only_cnt"], row["diff_sum"],
            )


# --------------------------------------------------------------------------- #
# read path
# --------------------------------------------------------------------------- #


def _level0_window(
    conn: sqlite3.Connection,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    period: int = CURRENT_PERIOD,
) -> list[dict]:
    """Read raw pixels; only the tiles intersecting the window are touched."""
    tx0, tx1 = x0 // TILE_PX, (x1 - 1) // TILE_PX
    ty0, ty1 = y0 // TILE_PX, (y1 - 1) // TILE_PX
    tiles: dict[tuple[int, int], list[list[int | None]]] = {}
    rows = conn.execute(
        """
        SELECT tile_x, tile_y, data FROM tiles
        WHERE tile_t = ? AND tile_x BETWEEN ? AND ?
                       AND tile_y BETWEEN ? AND ?
        """,
        (period, tx0, tx1, ty0, ty1),
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
    period: int = CURRENT_PERIOD,
) -> list[dict]:
    """Return every cell of ``level`` inside [x0,x1) x [y0,y1).

    Each row carries the valid original-pixel count and the mean coverage;
    the mean is null when no valid original pixel falls in the footprint.
    The read runs in one deferred transaction so a single query always sees
    one committed snapshot, even while another process is committing.
    """
    if not (0 <= level < LEVELS):
        raise ValueError(f"level must be within 0..{LEVELS - 1}")
    table = _summary_table(period)
    dim = GRID >> level
    if not (0 <= x0 < x1 <= dim and 0 <= y0 < y1 <= dim):
        raise ValueError(f"window must be within the {dim}x{dim} level grid")

    conn.execute("BEGIN")
    try:
        if level == 0:
            return _level0_window(conn, x0, y0, x1, y1, period)
        rows = conn.execute(
            f"""
            SELECT cx, cy, cnt, total FROM {table}
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


def query_change_window(
    conn: sqlite3.Connection,
    level: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[dict]:
    """Compare the two periods over a window using the same coordinates.

    For levels 1..10 each cell reports exact original-pixel counts:
    ``both_count`` (valid in both periods), ``baseline_only_count`` and
    ``current_only_count``.  ``mean_change_pp`` is the average
    ``current - baseline`` over paired pixels in percentage points, or null
    when the cell has no paired pixel.

    Only ``change_summaries`` is read, in one deferred snapshot transaction,
    so the whole window comes from a single committed state.
    """
    if not (1 <= level < LEVELS):
        raise ValueError("change queries support levels 1..%d" % (LEVELS - 1))
    dim = GRID >> level
    if not (0 <= x0 < x1 <= dim and 0 <= y0 < y1 <= dim):
        raise ValueError(f"window must be within the {dim}x{dim} level grid")

    conn.execute("BEGIN")
    try:
        rows = conn.execute(
            """
            SELECT cx, cy, both_cnt, baseline_only_cnt, current_only_cnt,
                   diff_sum
            FROM change_summaries
            WHERE level = ? AND cx BETWEEN ? AND ?
                          AND cy BETWEEN ? AND ?
            """,
            (level, x0, x1 - 1, y0, y1 - 1),
        ).fetchall()
        stored = {
            (r["cx"], r["cy"]): (
                r["both_cnt"],
                r["baseline_only_cnt"],
                r["current_only_cnt"],
                r["diff_sum"],
            )
            for r in rows
        }
        out: list[dict] = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                both, baseline_only, current_only, diff_sum = stored.get(
                    (x, y), (0, 0, 0, 0)
                )
                change = round(diff_sum / both, 4) if both else None
                out.append(
                    {
                        "x": x,
                        "y": y,
                        "both_count": both,
                        "baseline_only_count": baseline_only,
                        "current_only_count": current_only,
                        "mean_change_pp": change,
                    }
                )
        return out
    finally:
        conn.execute("COMMIT")


def get_tile(
    conn: sqlite3.Connection,
    tile_x: int,
    tile_y: int,
    period: int = CURRENT_PERIOD,
) -> dict | None:
    """Fetch one period's tile as the submitted 16x16 matrix, or None."""
    if period not in PERIODS:
        raise ValueError("period must be current (0) or baseline (1)")
    if not (0 <= tile_x < TILES_PER_AXIS and 0 <= tile_y < TILES_PER_AXIS):
        raise ValueError(
            f"tile coordinates must be within 0..{TILES_PER_AXIS - 1}"
        )
    row = conn.execute(
        """
        SELECT data, updated_at FROM tiles
        WHERE tile_t = ? AND tile_x = ? AND tile_y = ?
        """,
        (period, tile_x, tile_y),
    ).fetchone()
    if row is None:
        return None
    return {
        "period": PERIOD_NAMES[period],
        "tile_x": tile_x,
        "tile_y": tile_y,
        "values": decode_tile(row["data"]),
        "updated_at": row["updated_at"],
    }

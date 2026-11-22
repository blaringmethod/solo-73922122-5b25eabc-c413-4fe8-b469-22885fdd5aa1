"""FastAPI service: corrected canopy-cover raster, tile replacement and
multi-scale window queries backed by a single shared SQLite database.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from . import db
from .pool import ConnectionPool, Entry

DB_PATH = os.environ.get("CANOPY_DB_PATH", "/data/canopy.db")

# A level-0 window is bounded so a query cannot pull the whole 1M-cell grid
# in one response; coarse levels are small enough to stay unrestricted.
MAX_LEVEL0_CELLS = 262_144  # e.g. 512x512 raw pixels

_pool: ConnectionPool | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pool
    db.init_db(DB_PATH)
    _pool = ConnectionPool(DB_PATH)
    yield
    _pool.close_all()


app = FastAPI(
    title="Canopy Cover Raster Service",
    description=(
        "1024x1024 equal-area canopy-coverage raster. Submit corrections as "
        "16x16 tile replacements (integers 0-100 or null); adjacent 2x2 cells "
        "are aggregated into coarser levels 1..10. Query any level/window for "
        "valid original-pixel counts and mean coverage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TilePayload(BaseModel):
    values: list[list[Annotated[int, Field(ge=0, le=100)] | None]] = Field(
        description="16 rows of 16 cells; each cell is 0..100 or null."
    )

    @field_validator("values")
    @classmethod
    def _check_shape(cls, v: list[list]) -> list[list]:
        if len(v) != db.TILE_PX or any(
            len(row) != db.TILE_PX for row in v
        ):
            raise ValueError(f"payload must be {db.TILE_PX}x{db.TILE_PX}")
        return v


def get_entry():
    """Check out one connection (+lock) for the whole request, then return it."""
    assert _pool is not None
    entry = _pool.acquire()
    try:
        yield entry
    finally:
        _pool.release(entry)


EntryDep = Annotated[Entry, Depends(get_entry)]


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "canopy-cover-raster",
        "grid": db.GRID,
        "tile_px": db.TILE_PX,
        "tiles_per_axis": db.TILES_PER_AXIS,
        "levels": db.LEVELS,
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health(entry: EntryDep) -> dict:
    entry.conn.execute("SELECT 1").fetchone()
    return {"status": "ok"}


@app.put("/tiles/{tile_x}/{tile_y}", tags=["write"])
def replace_tile(
    tile_x: int, tile_y: int, payload: TilePayload, entry: EntryDep
) -> dict:
    """Replace a whole 16x16 tile; only affected pyramid summaries are updated.

    Atomic: a failure keeps the previous tile and every summary unchanged.
    """
    try:
        db.replace_tile(
            entry.conn,
            tile_x,
            tile_y,
            payload.values,
            datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.OperationalError as exc:
        # busy/lock timeout against another writer (e.g. the second instance)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "replaced",
        "tile_x": tile_x,
        "tile_y": tile_y,
        "cells": db.TILE_PX * db.TILE_PX,
    }


@app.get("/tiles/{tile_x}/{tile_y}", tags=["read"])
def read_tile(tile_x: int, tile_y: int, entry: EntryDep) -> dict:
    """Return the last submitted 16x16 matrix (404 if the tile was never set)."""
    try:
        result = db.get_tile(entry.conn, tile_x, tile_y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="tile has no committed data yet (all pixels null)",
        )
    return result


@app.get("/raster", tags=["read"])
def read_raster(
    entry: EntryDep,
    level: int = Query(0, ge=0, le=db.LEVELS - 1, description="0 = raw 1024 grid"),
    x0: int = Query(..., ge=0, description="window start x (inclusive)"),
    y0: int = Query(..., ge=0, description="window start y (inclusive)"),
    x1: int = Query(..., ge=1, description="window end x (exclusive)"),
    y1: int = Query(..., ge=1, description="window end y (exclusive)"),
) -> dict:
    """Query a window at one scale.

    Each returned cell gives ``valid_count`` (number of valid *original*
    pixels in its footprint) and ``mean`` coverage; ``mean`` is null when no
    valid original pixel exists.  One query always observes a single committed
    state.
    """
    dim = db.GRID >> level
    if not (0 <= x0 < x1 <= dim and 0 <= y0 < y1 <= dim):
        raise HTTPException(
            status_code=400,
            detail=f"window out of range for level {level} ({dim}x{dim} grid)",
        )
    if level == 0 and (x1 - x0) * (y1 - y0) > MAX_LEVEL0_CELLS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"level-0 window exceeds {MAX_LEVEL0_CELLS} cells; query a "
                "smaller window or a coarser level"
            ),
        )
    cells = db.query_window(entry.conn, level, x0, y0, x1, y1)
    return {
        "level": level,
        "window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "grid_size": dim,
        "cells": cells,
    }

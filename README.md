# Canopy Cover Raster Service

A FastAPI + SQLite service that stores a corrected **1024×1024 equal-area
canopy-cover raster** (initially entirely null) and serves it at multiple
aggregation scales.

## Data model

* The level-0 grid is 1024×1024 integer coverage values (0–100) or null,
  partitioned into **64×64 tiles of 16×16 pixels**. Corrections are submitted
  one whole tile at a time (`PUT /tiles/{x}/{y}`) and replace the tile
  entirely.
* Adjacent 2×2 cells are merged into one coarser cell, producing levels
  **1 (512) … 10 (1)**. Each coarse cell stores:
  * `valid_count` — number of valid *original* pixels in its footprint,
  * `mean` — their average coverage (`null` when `valid_count == 0`).
* Raw pixels live in `tiles` as signed int16 BLOBs (a dedicated sentinel
  marks nulls); aggregates live in `summaries` as `(count, total)` pairs so
  means never need a re-scan of raw pixels.

## Concurrency and durability guarantees

* Every tile replacement runs in a single `BEGIN IMMEDIATE` transaction that
  rewrites the tile and refreshes **only the pyramid ancestors it touches**
  (level 1 from raw pixels; higher levels from child summary rows — raw
  pixels of other tiles are never read).
* Two API instances sharing one database file therefore serialise on the
  SQLite write lock; concurrent replacements of different tiles under the
  same coarse cell are both accumulated (no lost update).
* Readers run in a snapshot transaction (WAL mode), so one query always sees
  a single committed state, never a half-updated pyramid.
* Any failure rolls the whole replacement back: the previous tile and all
  summaries stay intact, and data survives a process restart.
* The web layer uses a checked-out connection pool (`check_same_thread=False`
  plus a per-connection lock held for the whole request), which is safe with
  FastAPI's recycled worker threads.

## Run with Docker Compose

```bash
docker compose up -d --build
# two instances sharing one SQLite file:
#   http://localhost:8001/docs  (instance A)
#   http://localhost:8002/docs  (instance B)
```

The database is kept on the named volume `canopy-data` (`/data/canopy.db`),
so both containers share it and it survives restarts. Use a local/block
filesystem for the volume — SQLite locking is not reliable on NFS.

Run the self-test suite in a container:

```bash
docker compose --profile test run --rm selftest
```

## Run locally without Docker

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
export CANOPY_DB_PATH=/tmp/canopy.db
uvicorn app.main:app --reload
pytest -v
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `PUT` | `/tiles/{tile_x}/{tile_y}` | Replace one 16×16 tile (0–100 or null per cell) |
| `GET` | `/tiles/{tile_x}/{tile_y}` | Read the last submitted tile (404 if never written) |
| `GET` | `/raster?level=&x0=&y0=&x1=&y1=` | Window query; each cell returns `valid_count` and `mean` |
| `GET` | `/health` · `/` · `/docs` | Health check, metadata, interactive Swagger UI |

Tile coordinates range 0–63; raster windows use level-grid coordinates with
half-open ranges `[x0,x1) × [y0,y1)`. Level-0 windows are capped at 262 144
cells per request; use a coarser level for broad extents.

### Example

```bash
# write a tile whose cells are all 55
curl -X PUT localhost:8001/tiles/3/4 -H 'content-type: application/json' \
  -d '{"values": '"$(python -c 'import json;print(json.dumps([[55]*16]*16))')"'}'

# query the 8×8 level-1 cells covering that tile
curl 'localhost:8001/raster?level=1&x0=24&y0=32&x1=32&y1=40'
```

## Layout

```
app/db.py      SQLite schema, tile codec, replacement + pyramid refresh, queries
app/pool.py    request-scoped connection pool
app/main.py    FastAPI routes and validation
tests/         functional, concurrency (multi-process) and HTTP suites
```

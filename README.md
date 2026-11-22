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
* Two monitoring periods share the same 64×64 tile grid and payload format:
  `tile_t = 0` is the **current** period (all data written by the original
  service) and `tile_t = 1` is an optional **baseline** period. A period's
  unsubmitted tile is missing data (all pixels null).
* Each period has its own pyramid (`summaries`, `baseline_summaries`).
  `change_summaries` pairs the original pixels of both periods per cell and
  stores exact counts of `both`, `baseline_only`, `current_only` pixels plus
  the sum of `current - baseline` over paired pixels.

## Two-period comparison

`GET /change` compares the periods over the same window coordinates as
`GET /raster` (levels 1–10). Every cell reports:

| Field | Meaning |
| --- | --- |
| `both_count` | original pixels valid in **both** periods |
| `baseline_only_count` | original pixels valid only in the baseline |
| `current_only_count` | original pixels valid only in the current period |
| `mean_change_pp` | mean `current - baseline` over paired pixels, in percentage points; `null` when nothing is paired |

Counts are exact original-pixel counts at every level (they are summed from
level 1, not re-derived from coarse means).

## Concurrency and durability guarantees

* Every tile replacement runs in a single `BEGIN IMMEDIATE` transaction that
  rewrites the tile and refreshes **only the pyramid ancestors it touches**
  (level 1 from raw pixels; higher levels from child summary rows — raw
  pixels of other tiles are never read).
* The same transaction refreshes `change_summaries` for that tile position
  from the submitted payload plus the counterpart period's raw tile at
  **the same coordinates only**. Both corrections are committed together.
* Two API instances sharing one database file therefore serialise on the
  SQLite write lock; concurrent replacements of different tiles under the
  same coarse cell are both accumulated (no lost update), including one
  period corrected on each instance at the same tile.
* Readers run in a snapshot transaction (WAL mode), so one query always sees
  a single committed state, never a half-updated pyramid. `/change` reads only
  `change_summaries`, inside one such snapshot, so the whole window comes from
  the same committed state.
* Any failure rolls the whole replacement back: the previous tile and all
  summaries (including the comparison rows) stay intact, and data survives a
  process restart.
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
| `PUT` | `/tiles/{tile_x}/{tile_y}` | Replace one 16×16 **current-period** tile (original interface) |
| `GET` | `/tiles/{tile_x}/{tile_y}` | Read the last submitted current-period tile (404 if never written) |
| `PUT` | `/tiles/{period}/{tile_x}/{tile_y}` | Replace one tile of `current` or `baseline` at the same coordinates |
| `GET` | `/tiles/{period}/{tile_x}/{tile_y}` | Read one period's tile (404 if never written) |
| `GET` | `/raster?level=&x0=&y0=&x1=&y1=[&period=]` | Window query; each cell returns `valid_count` and `mean` |
| `GET` | `/change?level=&x0=&y0=&x1=&y1=` | Two-period comparison per cell (levels 1–10) |
| `GET` | `/health` · `/` · `/docs` | Health check, metadata, interactive Swagger UI |

Tile coordinates range 0–63; raster windows use level-grid coordinates with
half-open ranges `[x0,x1) × [y0,y1)`. Level-0 windows are capped at 262 144
cells per request; use a coarser level for broad extents.

### Example

```bash
# current period: tile whose cells are all 55 (original 3-part endpoint)
curl -X PUT localhost:8001/tiles/3/4 -H 'content-type: application/json' \
  -d '{"values": '"$(python -c 'import json;print(json.dumps([[55]*16]*16))')"'}'

# baseline period: same coordinates and payload, all 45
curl -X PUT localhost:8001/tiles/baseline/3/4 -H 'content-type: application/json' \
  -d '{"values": '"$(python -c 'import json;print(json.dumps([[45]*16]*16))')"'}'

# query the 8×8 level-1 cells covering that tile: 4 paired pixels per cell,
# mean change +10 pp
curl 'localhost:8001/change?level=1&x0=24&y0=32&x1=32&y1=40'

# the current raster is unchanged
curl 'localhost:8001/raster?level=1&x0=24&y0=32&x1=32&y1=40'
```

Databases created by the original single-period service are migrated on
startup: existing corrections remain the current period, and the comparison
summaries are backfilled once.

## Layout

```
app/db.py      SQLite schema, tile codec, replacement + pyramid/change refresh,
               queries
app/pool.py    request-scoped connection pool
app/main.py    FastAPI routes and validation
tests/         functional, two-period, concurrency (multi-process) and HTTP
               suites
```

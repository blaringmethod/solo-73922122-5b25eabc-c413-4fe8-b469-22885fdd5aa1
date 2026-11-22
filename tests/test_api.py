"""End-to-end HTTP tests via FastAPI TestClient, including threaded
concurrent replacements sharing one database file."""

import concurrent.futures

from tests.conftest import make_tile


def test_docs_and_meta(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["grid"] == 1024 and body["levels"] == 11

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_initial_query_all_null(client):
    r = client.get("/raster", params={"level": 3, "x0": 0, "y0": 0,
                                      "x1": 128, "y1": 128})
    assert r.status_code == 200
    cells = r.json()["cells"]
    assert len(cells) == 128 * 128
    assert all(c["valid_count"] == 0 and c["mean"] is None for c in cells)


def test_put_then_get_tile_and_raster(client):
    r = client.put("/tiles/3/4", json={"values": make_tile(55)})
    assert r.status_code == 200 and r.json()["status"] == "replaced"

    r = client.get("/tiles/3/4")
    assert r.status_code == 200
    assert r.json()["values"][15][15] == 55

    r = client.get("/raster", params={"level": 1, "x0": 24, "y0": 32,
                                      "x1": 32, "y1": 40})
    cells = r.json()["cells"]
    assert len(cells) == 64
    assert all(c["valid_count"] == 4 and c["mean"] == 55.0 for c in cells)


def test_tile_never_written_returns_404(client):
    r = client.get("/tiles/63/63")
    assert r.status_code == 404


def test_validation_errors(client):
    bad_shape = {"values": [[1] * 15] * 16}
    assert client.put("/tiles/0/0", json=bad_shape).status_code == 422

    bad_value = {"values": make_tile(101)}
    assert client.put("/tiles/0/0", json=bad_value).status_code == 422

    assert client.put("/tiles/64/0", json={"values": make_tile(1)}).status_code == 400

    r = client.get("/raster", params={"level": 0, "x0": 0, "y0": 0,
                                      "x1": 1025, "y1": 1})
    assert r.status_code == 400

    r = client.get("/raster", params={"level": 0, "x0": 0, "y0": 0,
                                      "x1": 512, "y1": 513})
    assert r.status_code == 413  # level-0 window cap


def test_concurrent_http_replacements_same_coarse_cell(client):
    # Two threads act like two API instances sharing the database file.
    def put(tx, value):
        return client.put(f"/tiles/{tx}/0", json={"values": make_tile(value)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(put, 0, 10), pool.submit(put, 1, 90)]
        results = [f.result() for f in futs]
    assert all(r.status_code == 200 for r in results)

    r = client.get("/raster", params={"level": 5, "x0": 0, "y0": 0,
                                      "x1": 1, "y1": 1})
    cell = r.json()["cells"][0]
    assert cell["valid_count"] == 512
    assert cell["mean"] == 50.0  # (256*10 + 256*90) / 512


def test_null_payload_round_trip(client):
    client.put("/tiles/10/10", json={"values": make_tile(None)})
    r = client.get("/tiles/10/10")
    assert r.status_code == 200
    assert r.json()["values"][0][0] is None

    r = client.get("/raster", params={"level": 1, "x0": 80, "y0": 80,
                                      "x1": 88, "y1": 88})
    assert all(c["mean"] is None and c["valid_count"] == 0
               for c in r.json()["cells"])

"""HTTP-level tests for the two-period comparison API."""

import concurrent.futures

from tests.conftest import make_tile, partial_tile


def test_original_current_endpoints_unchanged(client):
    r = client.put("/tiles/3/4", json={"values": make_tile(55)})
    assert r.status_code == 200 and r.json()["status"] == "replaced"

    assert client.get("/tiles/3/4").json()["values"][0][0] == 55
    r = client.get("/raster", params={"level": 1, "x0": 24, "y0": 32,
                                      "x1": 32, "y1": 40})
    assert all(c["mean"] == 55.0 and c["valid_count"] == 4
               for c in r.json()["cells"])


def test_submit_both_periods_and_query_change(client):
    assert client.put(
        "/tiles/current/3/4", json={"values": make_tile(75)}
    ).status_code == 200
    assert client.put(
        "/tiles/baseline/3/4", json={"values": make_tile(50)}
    ).status_code == 200

    # baseline was submitted under the same coordinates and format
    assert client.get("/tiles/baseline/3/4").json()["values"][0][0] == 50
    assert client.get(
        "/tiles/current/3/4"
    ).json()["values"][0][0] == 75

    r = client.get("/change", params={"level": 1, "x0": 24, "y0": 32,
                                      "x1": 32, "y1": 40})
    assert r.status_code == 200
    body = r.json()
    assert body["level"] == 1 and len(body["cells"]) == 64
    for cell in body["cells"]:
        assert cell["both_count"] == 4
        assert cell["baseline_only_count"] == 0
        assert cell["current_only_count"] == 0
        assert cell["mean_change_pp"] == 25.0

    # the current raster still reports current-period values
    cur = client.get("/raster", params={"level": 1, "x0": 24, "y0": 32,
                                        "x1": 32, "y1": 40})
    assert all(c["mean"] == 75.0 for c in cur.json()["cells"])
    base = client.get("/raster", params={"level": 1, "x0": 24, "y0": 32,
                                         "x1": 32, "y1": 40,
                                         "period": "baseline"})
    assert all(c["mean"] == 50.0 for c in base.json()["cells"])


def test_unsubmitted_baseline_is_missing(client):
    client.put("/tiles/2/3", json={"values": partial_tile(10).copy()})
    r = client.get("/change", params={"level": 5, "x0": 1, "y0": 1,
                                      "x1": 2, "y1": 2})
    cell = r.json()["cells"][0]
    assert cell["both_count"] == 0
    assert cell["baseline_only_count"] == 0
    assert cell["current_only_count"] == 128
    assert cell["mean_change_pp"] is None


def test_change_validation(client):
    base = {"x0": 0, "y0": 0, "x1": 1, "y1": 1}
    assert client.get(
        "/change", params={**base, "level": 0}
    ).status_code == 422
    assert client.get(
        "/change", params={**base, "level": 11}
    ).status_code == 422
    assert client.get(
        "/change", params={"level": 1, "x0": 0, "y0": 0, "x1": 600, "y1": 1}
    ).status_code == 400
    assert client.put(
        "/tiles/monthly/0/0", json={"values": make_tile(1)}
    ).status_code == 422
    assert client.put(
        "/tiles/baseline/64/0", json={"values": make_tile(1)}
    ).status_code == 400
    assert client.get("/tiles/baseline/63/63").status_code == 404


def test_concurrent_http_corrections_both_periods(client):
    # two API instances (threads) correcting both periods at the same tile
    def put(period, value):
        return client.put(
            f"/tiles/{period}/0/0", json={"values": make_tile(value)}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(put, "baseline", 30),
                pool.submit(put, "current", 85)]
        results = [f.result() for f in futs]
    assert all(r.status_code == 200 for r in results)

    cell = client.get(
        "/change", params={"level": 5, "x0": 0, "y0": 0, "x1": 1, "y1": 1}
    ).json()["cells"][0]
    assert cell["both_count"] == 256
    assert cell["mean_change_pp"] == 55.0


def test_second_instance_commit_visible_to_first(client, db_path):
    """A second process/connection on the same file is reflected by /change."""
    from app import db

    client.put("/tiles/baseline/1/1", json={"values": make_tile(20)})

    # second instance: its own connection to the shared database file
    other = db.connect(db_path)
    try:
        db.replace_tile(
            other, 1, 1, make_tile(60), "second-instance",
            db.CURRENT_PERIOD,
        )
    finally:
        other.close()

    cell = client.get(
        "/change", params={"level": 5, "x0": 0, "y0": 0, "x1": 1, "y1": 1}
    ).json()["cells"][0]
    assert cell["both_count"] == 256
    assert cell["mean_change_pp"] == 40.0

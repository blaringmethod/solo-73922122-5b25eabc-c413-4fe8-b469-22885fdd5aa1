import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "canopy-test.db")


@pytest.fixture()
def conn(db_path):
    db.init_db(db_path)
    connection = db.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture()
def client(db_path, monkeypatch):
    # point the app's dependency at this test's isolated database
    monkeypatch.setattr("app.main.DB_PATH", db_path)
    with TestClient(app) as c:
        yield c


def make_tile(fill, null_mask=None):
    """16x16 tile filled with `fill`; null_mask is a set of (y, x)."""
    null_mask = null_mask or set()
    return [
        [None if (y, x) in null_mask else fill for x in range(16)]
        for y in range(16)
    ]


def partial_tile(even_value=10, odd_value=None):
    return [
        [even_value if (x + y) % 2 == 0 else odd_value for x in range(16)]
        for y in range(16)
    ]

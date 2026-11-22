"""SQLite connection pool with per-request checkout semantics.

FastAPI runs sync endpoints on a recycled worker-thread pool, so a plain
``threading.local`` connection is not safe: when an OS thread is recycled, two
requests can share one connection while the first is mid-transaction, and a
lock that only guards individual ``execute()`` calls cannot keep a whole
BEGIN..COMMIT transaction atomic across that handoff.

This pool instead lends a *connection + its lock* for the duration of one
request:

* ``acquire()`` blocks until a free connection is available (or creates one),
  and takes that connection's exclusive lock;
* ``release()`` unlocks and returns it to the pool.

While held, no other request can touch the connection, so a multi-statement
transaction stays on one consistent owner regardless of thread recycling.
Connections are opened with ``check_same_thread=False`` because the owning
request may resume on a different worker thread after an await boundary.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field

from . import db


@dataclass
class Entry:
    conn: sqlite3.Connection
    lock: threading.Lock = field(default_factory=threading.Lock)


class ConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 32) -> None:
        self._db_path = db_path
        self._max = max_connections
        self._idle: list[Entry] = []
        self._created = 0
        self._pool_lock = threading.Lock()
        self._available = threading.Semaphore(max_connections)

    def acquire(self) -> Entry:
        # Block while every connection is checked out.
        self._available.acquire()
        with self._pool_lock:
            if self._idle:
                entry = self._idle.pop()
            else:
                self._created += 1
                conn = db.connect(self._db_path, check_same_thread=False)
                entry = Entry(conn=conn)
        # Take this connection's exclusive lock for the whole request; it is
        # released only when the dependency calls release() afterwards.
        entry.lock.acquire()
        return entry

    def release(self, entry: Entry) -> None:
        with self._pool_lock:
            self._idle.append(entry)
        entry.lock.release()
        self._available.release()

    def close_all(self) -> None:
        with self._pool_lock:
            for entry in self._idle:
                entry.conn.close()
            self._idle.clear()

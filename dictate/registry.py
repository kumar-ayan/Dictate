from __future__ import annotations
import sqlite3
from .util import ROOT

DB = ROOT / "registry.db"

def connect() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS shards(name TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER, hours REAL, schema_json TEXT, stats_json TEXT, consumed_asr INTEGER DEFAULT 0, consumed_cleanup INTEGER DEFAULT 0, added_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, kind TEXT, record_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS dictations(id INTEGER PRIMARY KEY, raw TEXT, cleaned TEXT, final TEXT, timings_json TEXT, model_versions TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    c.commit()
    return c

def set_meta(key: str, value: str) -> None:
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (key, value))

def get_meta(key: str, default: str | None = None) -> str | None:
    with connect() as c:
        r = c.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

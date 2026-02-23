"""
SQLite persistence layer.

Tables we manage:
  sessions    — one row per conversation thread
  conversions — one row per PDF converted

LangGraph's SqliteSaver manages its own tables (checkpoints, etc.)
in the same memory.db file automatically.
"""

import sqlite3
from datetime import datetime, timezone
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "memory.db"


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver connected to memory.db (used as context manager in main)."""
    return SqliteSaver.from_conn_string(DB_PATH)


def init_db() -> None:
    """Create our custom tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id      TEXT PRIMARY KEY,
            created_at     TEXT NOT NULL,
            last_active    TEXT NOT NULL,
            message_count  INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id     TEXT NOT NULL,
            pdf_file      TEXT NOT NULL,
            csv_file      TEXT NOT NULL,
            processed_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def upsert_session(thread_id: str, delta_messages: int = 0) -> None:
    """Insert a new session or update last_active + message_count."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO sessions (thread_id, created_at, last_active, message_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            last_active   = excluded.last_active,
            message_count = message_count + excluded.message_count
    """, (thread_id, now, now, delta_messages))
    conn.commit()
    conn.close()


def log_conversion(thread_id: str, pdf_file: str, csv_file: str) -> None:
    """Record a completed PDF → CSV conversion."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO conversions (thread_id, pdf_file, csv_file, processed_at) VALUES (?, ?, ?, ?)",
        (thread_id, pdf_file, csv_file, now),
    )
    conn.commit()
    conn.close()


def list_sessions() -> list[dict]:
    """Return the 10 most recently active sessions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT thread_id, created_at, last_active, message_count
        FROM sessions
        ORDER BY last_active DESC
        LIMIT 10
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_conversions(thread_id: str | None = None) -> list[dict]:
    """Return conversions, optionally filtered by thread."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if thread_id:
        rows = conn.execute(
            "SELECT * FROM conversions WHERE thread_id = ? ORDER BY processed_at DESC",
            (thread_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM conversions ORDER BY processed_at DESC LIMIT 20"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

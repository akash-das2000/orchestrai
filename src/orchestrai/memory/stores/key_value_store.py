# src/orchestrai/memory/stores/key_value_store.py

import sqlite3
from typing import Optional

class KeyValueStore:
    """
    A simple key/value store backed by SQLite.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Opens (or creates) the SQLite DB at db_path.
        If db_path=":memory:", uses an in-memory database.
        """
        self.conn = sqlite3.connect(db_path)
        # Ensure the table exists
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def set(self, key: str, value: str) -> None:
        """
        Insert or update the given key with value.
        """
        self.conn.execute(
            "REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve the value for a key, or None if missing.
        """
        cur = self.conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def delete(self, key: str) -> None:
        """
        Remove a key (no-op if key not present).
        """
        self.conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        self.conn.commit()

    def keys(self) -> list[str]:
        """
        List all keys currently stored.
        """
        cur = self.conn.execute("SELECT key FROM kv_store")
        return [row[0] for row in cur.fetchall()]

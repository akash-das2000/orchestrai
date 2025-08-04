# tests/test_key_value_store.py

import os
import sqlite3
import pytest
from orchestrai.memory.stores.key_value_store import KeyValueStore

def test_kv_store_crud(tmp_path):
    # Use a real file under tmp_path for persistence
    db_file = tmp_path / "test_kv.db"
    store = KeyValueStore(str(db_file))

    # Initially empty
    assert store.get("foo") is None
    assert store.keys() == []

    # Create
    store.set("foo", "bar")
    assert store.get("foo") == "bar"
    assert store.keys() == ["foo"]

    # Update
    store.set("foo", "baz")
    assert store.get("foo") == "baz"
    assert store.keys() == ["foo"]

    # Insert another
    store.set("hello", "world")
    ks = store.keys()
    assert set(ks) == {"foo", "hello"}

    # Delete
    store.delete("foo")
    assert store.get("foo") is None
    assert store.keys() == ["hello"]

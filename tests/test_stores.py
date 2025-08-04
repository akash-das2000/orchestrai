# tests/test_stores.py

import pytest
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore

def test_rolling_buffer_basic():
    store = RollingBufferStore(max_size=3)
    store.add("k1", "v1")
    store.add("k2", "v2")
    store.add("k3", "v3")
    # Should return the last two entries
    assert store.query("", top_k=2) == [
        ("k2", "v2", {}),
        ("k3", "v3", {}),
    ]

def test_rolling_buffer_overflow():
    store = RollingBufferStore(max_size=2)
    store.add("k1", "v1")
    store.add("k2", "v2")
    store.add("k3", "v3")  # k1 is dropped
    # Even if we ask for top_k=3, only 2 items exist
    assert store.query("", top_k=3) == [
        ("k2", "v2", {}),
        ("k3", "v3", {}),
    ]

def test_query_less_than_buffer_size():
    store = RollingBufferStore(max_size=5)
    store.add("x", "100")
    # Asking for more than exists should just return what we have
    assert store.query("", top_k=4) == [
        ("x", "100", {}),
    ]

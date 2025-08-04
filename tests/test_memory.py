# tests/test_memory.py

from orchestrai.memory.core import MemoryStore

class DummyMemory(MemoryStore):
    def __init__(self):
        self.data = []

    def add(self, key, value, metadata=None):
        # simply record the triple
        self.data.append((key, value, metadata or {}))

    def query(self, query, top_k=5):
        # return the last top_k items
        return self.data[-top_k:]

    def summarize(self):
        # no-op for this dummy
        pass

def test_dummy_memory():
    m = DummyMemory()
    m.add("a", "1")
    m.add("b", "2")
    # expect exactly the two we just added
    assert m.query("", top_k=2) == [("a","1",{}), ("b","2",{})]

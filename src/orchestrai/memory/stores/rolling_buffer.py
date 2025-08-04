# src/orchestrai/memory/stores/rolling_buffer.py

from collections import deque
from typing import List, Tuple, Dict

from orchestrai.memory.core import MemoryStore

class RollingBufferStore(MemoryStore):
    """
    A fixed-size in-memory buffer: keeps only the last `max_size` entries.
    """

    def __init__(self, max_size: int = 50):
        self.buffer = deque(maxlen=max_size)

    def add(self, key: str, value: str, metadata: Dict = None) -> None:
        """
        Append a new (key, value, metadata) tuple.
        Oldest entries drop off when capacity is exceeded.
        """
        self.buffer.append((key, value, metadata or {}))

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        """
        Return the last `top_k` items in insertion order.
        (We ignore `query` for this simple strategy.)
        """
        items = list(self.buffer)
        return items[-top_k:]

    def summarize(self) -> None:
        """
        No-op: this store doesn’t support summarization.
        """
        pass

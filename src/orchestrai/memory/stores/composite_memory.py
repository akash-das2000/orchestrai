# src/orchestrai/memory/stores/composite_memory.py

from typing import List, Tuple, Dict, Any, Optional
from orchestrai.memory.core import MemoryStore


class CompositeMemoryStore(MemoryStore):
    def __init__(
        self,
        recency_store: MemoryStore,
        semantic_store: MemoryStore,
        kv_store: Optional[Any] = None,
    ):
        self.recency = recency_store
        self.semantic = semantic_store
        self.kv = kv_store

    def add(self, key: str, value: str, metadata: Dict[str, Any] = None) -> None:
        self.recency.add(key, value, metadata)
        self.semantic.add(key, value, metadata)

    def query_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Retrieve top-k semantically relevant memory entries."""
        results = []
        raw = self.semantic.query(query, top_k)
        for key, meta in raw:
            content = meta.get("text", "")
            results.append((key, content, meta or {}))
        return results

    def query_recency(self, top_k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Retrieve top-k most recent memory entries."""
        return self.recency.query("", top_k)

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_semantic: bool = True,
        use_recency: bool = True
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Retrieve both semantic + recency entries, de-duplicated by key.
        """
        merged = []
        seen_keys = set()

        if use_semantic:
            for key, content, meta in self.query_semantic(query, top_k):
                if key not in seen_keys:
                    merged.append((key, content, meta))
                    seen_keys.add(key)

        if use_recency:
            for key, content, meta in self.query_recency(top_k):
                if key not in seen_keys:
                    merged.append((key, content, meta))
                    seen_keys.add(key)

        return merged

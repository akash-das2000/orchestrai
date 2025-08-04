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

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Returns a list of (role, content, metadata) triplets,
        merging semantic & recency stores without duplicates.
        """
        # 1) Get semantic hits: they come back as (key, meta)
        raw_sem = self.semantic.query(query, top_k)
        sem_triplets: List[Tuple[str, str, Dict[str, Any]]] = []
        for key, meta in raw_sem:
            # extract the content we stored under 'text'
            content = meta.get("text", "")
            sem_triplets.append((key, content, meta or {}))

        # 2) Get recency hits: they already come as (role, content, meta)
        rec_triplets = self.recency.query("", top_k)

        # 3) Merge, de-duplicate by key (first-occurrence wins)
        merged: List[Tuple[str, str, Dict[str, Any]]] = []
        seen = set()
        for entry in (sem_triplets + rec_triplets):
            role = entry[0]
            if role in seen:
                continue
            seen.add(role)
            merged.append(entry)
            if len(merged) >= top_k:
                break

        return merged

    def summarize(self) -> None:
        if hasattr(self.recency, "summarize"):
            self.recency.summarize()

    def kv_set(self, key: str, value: str) -> None:
        if self.kv:
            self.kv.set(key, value)

    def kv_get(self, key: str) -> Optional[str]:
        return self.kv.get(key) if self.kv else None

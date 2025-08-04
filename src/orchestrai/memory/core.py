from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

class MemoryStore(ABC):
    """Base interface for any memory backend."""
    @abstractmethod
    def add(self, key: str, value: str, metadata: Dict = None) -> None:
        ...

    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        ...

    @abstractmethod
    def summarize(self) -> None:
        ...

import openai
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from orchestrai.memory.core import MemoryStore

class VectorMemoryStore(MemoryStore):
    """
    Semantic memory: embeds each (key,value) and indexes in a FAISS index.
    """

    def __init__(
        self,
        embed_model: str = "text-embedding-ada-002",
        dim: int = 1536,
    ):
        self.embed_model = embed_model
        # FAISS index for L2 similarity on float32 vectors
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Tuple[str, Dict[str, Any]]] = []

    def add(self, key: str, value: str, metadata: Dict[str, Any] = None) -> None:
        # 1) Embed the text
        resp = openai.embeddings.create(
            model=self.embed_model,
            input=value
        )
        vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

        # 2) Add vector to FAISS and store metadata
        self.index.add(vec)
        self.metadatas.append((key, metadata or {"content": value}))

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        # 1) Embed the query
        resp = openai.embeddings.create(
            model=self.embed_model,
            input=query
        )
        qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

        # 2) Search FAISS
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.metadatas):
                key, meta = self.metadatas[idx]
                results.append((key, meta))
        return results

    def summarize(self) -> None:
        # No-op for this store
        pass

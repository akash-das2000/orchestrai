import openai
from typing import List, Tuple, Dict, Any
from orchestrai.memory.core import MemoryStore

class SummarizingMemoryStore(MemoryStore):
    """
    Wraps another MemoryStore and auto-summarizes old entries when the 
    count exceeds a threshold.
    """
    def __init__(
        self,
        inner: MemoryStore,
        threshold: int = 20,
        chunk_size: int = 5,
        model: str = "gpt-4o-mini",
        summarizer_kwargs: Dict[str, Any] = None,
    ):
        self.inner = inner
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.model = model
        self.summarizer_kwargs = summarizer_kwargs or {
            "temperature": 0.3,
            # you can add max_tokens, top_p, etc.
        }

    def add(self, key: str, value: str, metadata: Dict = None) -> None:
        # 1) Add to inner store
        self.inner.add(key, value, metadata)

        # 2) If we exceed the threshold, summarize the oldest chunk_size entries
        entries = self.inner.query("", top_k=self.threshold + 1)
        if len(entries) > self.threshold:
            # Extract the oldest entries
            old_entries = entries[: self.chunk_size]
            text_to_summarize = "\n".join(
                f"{role}: {content}" for (role, content, _) in old_entries
            )

            # 3) Call OpenAI to summarize
            sys_msg = {
                "role": "system",
                "content": "You are a summarizer. Condense the following conversation into one concise statement:",
            }
            user_msg = {"role": "user", "content": text_to_summarize}
            resp = openai.chat.completions.create(
                model=self.model,
                messages=[sys_msg, user_msg],
                **self.summarizer_kwargs,
            )
            summary = resp.choices[0].message.content

            # 4) Remove the old entries from the inner buffer (if it’s a deque)
            if hasattr(self.inner, "buffer"):
                for _ in range(self.chunk_size):
                    self.inner.buffer.popleft()

            # 5) Add the summary back as a system message
            self.inner.add("system", summary)

    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        return self.inner.query(query, top_k)

    def summarize(self) -> None:
        # Expose manual summarization if needed
        pass

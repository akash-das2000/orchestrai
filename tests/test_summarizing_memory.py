import pytest
from unittest.mock import patch, MagicMock
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.stores.summarizing_memory import SummarizingMemoryStore

@patch("openai.chat.completions.create")
def test_summarizing_memory_basic(mock_create):
    # 1) Prepare a fake summary from OpenAI
    fake = MagicMock()
    fake.choices = [MagicMock(message=MagicMock(content="Condensed summary."))]
    mock_create.return_value = fake

    # 2) Inner buffer with plenty of capacity
    inner = RollingBufferStore(max_size=10)
    # Wrap it: summarize after >3 entries, 2 at a time
    store = SummarizingMemoryStore(inner, threshold=3, chunk_size=2, model="test", summarizer_kwargs={"temperature": 0})

    # 3) Add exactly threshold entries → no summary yet
    store.add("user", "u1")
    store.add("assistant", "a1")
    store.add("user", "u2")
    assert len(inner.buffer) == 3

    # 4) Add one more → triggers summarization
    store.add("assistant", "a2")

    # After summarization, buffer should have:
    # - the two newest turns (u2, a2)
    # - plus the summary as a system entry
    entries = inner.query("", top_k=5)
    roles_contents = [(r, c) for r, c, _ in entries]

    assert len(entries) == 3
    assert ("user", "u2") in roles_contents
    assert ("assistant", "a2") in roles_contents
    assert ("system", "Condensed summary.") in roles_contents

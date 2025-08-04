# tests/test_openai_adapter.py
import os
os.environ["OPENAI_API_KEY"] = "test"    # ← prevent OpenAIError in tests

import pytest
from unittest.mock import patch, MagicMock
import openai
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.adapters.openai_adapter import OpenAIAdapter

@patch("openai.chat.completions.create")
def test_openai_adapter_basic(mock_create):
    # Arrange: mock OpenAI response
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="OK!"))]
    mock_create.return_value = mock_resp

    store = RollingBufferStore(max_size=10)
    adapter = OpenAIAdapter(memory=store, model="test-model", default_chat_kwargs={"temperature": 0})

    # Act: call with a user message
    reply = adapter.call([{"role": "user", "content": "Hello"}])

    # Assert: reply returned and memory updated
    assert reply == "OK!"
    # The memory should contain the user message and assistant reply
    entries = store.query("", top_k=2)
    assert entries[0][:2] == ("user", "Hello")
    assert entries[1][:2] == ("assistant", "OK!")

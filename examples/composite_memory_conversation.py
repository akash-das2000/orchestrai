# examples/composite_memory_conversation.py

import os
import re
from dotenv import load_dotenv
import openai

from orchestrai.memory.stores.composite_memory import CompositeMemoryStore
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.stores.summarizing_memory import SummarizingMemoryStore
from orchestrai.memory.stores.vector_memory import VectorMemoryStore
from orchestrai.memory.stores.key_value_store import KeyValueStore
from orchestrai.memory.adapters.openai_adapter import OpenAIAdapter

def main():
    # 1) Load your OpenAI key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    # 2) Build sub-stores
    recency_buf = RollingBufferStore(max_size=20)
    summarizer = SummarizingMemoryStore(inner=recency_buf, threshold=5, chunk_size=2)
    semantic_store = VectorMemoryStore(embed_model="text-embedding-ada-002", dim=1536)
    kv_store = KeyValueStore(db_path="agent_memory.db")

    # 3) Compose them
    composite = CompositeMemoryStore(
        recency_store=summarizer,
        semantic_store=semantic_store,
        kv_store=kv_store
    )

    # 4) Adapter wrapping the composite memory
    adapter = OpenAIAdapter(
        memory=composite,
        model="gpt-4o-mini",
        default_chat_kwargs={"temperature": 0.5, "max_tokens": 100}
    )

    # 5) A mini-conversation
    dialogue = [
        "Hello, who am I?",                # regular LLM turn
        "Remember: my name is Akash.",     # KV->store
        "Hi again! What's my name?",       # KV lookup
        "What did I ask first?",           # semantic/recency recall
        "Tell me a fun fact about trees."  # generic LLM turn
    ]

    for turn in dialogue:
        print(f"\nUser: {turn}")

        # If it's a "Remember" command, store it in KV directly
        m = re.match(r"Remember:\s*my name is (.+)", turn, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            composite.kv_set("user.name", name)
            response = f"Okay, I'll remember your name is {name}."
        else:
            # Build messages list: inject stored user.name as a system message
            messages = []
            name = composite.kv_get("user.name")
            if name:
                messages.append({"role": "system", "content": f"User’s name is {name}"})
            messages.append({"role": "user", "content": turn})

            # Call the adapter (does real API call + memory save)
            response = adapter.call(messages)

        print(f"Assistant: {response}")

    # 6) Show explicit KV fact
    print("\n— Explicit KV fact —")
    print("user.name =", composite.kv_get("user.name"))

    # 7) Semantic recall for "first"
    print("\n— Semantic recall for 'first' —")
    hits = composite.query("first", top_k=3)
    for i, (key, content, meta) in enumerate(hits, start=1):
        text = meta.get("text", content)
        print(f"  {i}. key={key}, text={text}")

    # 8) Recency buffer contents
    print("\n— Recency buffer —")
    for role, content, _ in recency_buf.query("", top_k=10):
        print(f"  [{role}] {content}")

if __name__ == "__main__":
    main()

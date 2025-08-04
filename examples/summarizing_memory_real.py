# examples/summarizing_memory_real.py

import os
from dotenv import load_dotenv
import openai

# 1) Load API key
load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")

from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.stores.summarizing_memory import SummarizingMemoryStore

def print_memory(store):
    entries = store.query("", top_k=20)
    if not entries:
        print("  (empty)")
    else:
        for i, (role, content, _) in enumerate(entries, 1):
            print(f"  {i}. [{role}] {content}")

def main():
    # 2) Wrap a rolling buffer with summarization:
    #    threshold=3 entries, chunk_size=2 (so every 4th add triggers summarization of the oldest 2)
    inner = RollingBufferStore(max_size=10)
    store = SummarizingMemoryStore(
        inner,
        threshold=3,
        chunk_size=2,
        model="gpt-4o-mini",
        summarizer_kwargs={"temperature": 0.3, "max_tokens": 50}
    )

    # 3) Simulate a conversation
    prompts = [
        "Hi, what's your name?",
        "Tell me about yourself.",
        "What's the weather like today?",
        "Can you summarize what I asked so far?",
        "And now, what did I ask first?"
    ]

    for round_idx, text in enumerate(prompts, start=1):
        print(f"\n=== Round {round_idx} ===")
        print("User:", text)

        # show memory before
        print("Memory before call:")
        print_memory(inner)

        # call summarizing store directly
        store.add("user", text)

        # for this example we’ll just ask the store itself to summarize directly via OpenAI
        # (so that it adds the “assistant” turn too)
        # wrap that in another add
        # Note: in real use you’d feed through your adapter, but this illustrates summarization.
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":text}
            ],
            temperature=0.5,
            max_tokens=60,
        ).choices[0].message.content
        store.add("assistant", resp)
        print("Assistant:", resp)

        # show memory after
        print("Memory after call:")
        print_memory(inner)

if __name__ == "__main__":
    main()

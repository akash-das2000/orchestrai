# examples/memory_real_verbose.py

import os
from dotenv import load_dotenv

# 1) Load your .env keys
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.adapters.openai_adapter import OpenAIAdapter

def show_memory(mem: RollingBufferStore):
    entries = mem.query("", top_k=100)
    if not entries:
        print("  (memory is empty)")
    else:
        for idx, (role, content, meta) in enumerate(entries, 1):
            print(f"  {idx}. [{role}] {content}")

def main():
    # 2) Build memory + adapter
    mem = RollingBufferStore(max_size=10)
    adapter = OpenAIAdapter(
        memory=mem,
        model="gpt-4o-mini",
        default_chat_kwargs={"temperature": 0.5, "max_tokens": 100}
    )

    # 3) A sequence of user prompts
    user_messages = [
        "Hello, who are you?",
        "Can you tell me a joke?",
        "What did I ask you first?"
    ]

    for round_idx, text in enumerate(user_messages, start=1):
        print(f"\n=== Round {round_idx} ===")
        print(">>> User:", text)

        # 4) Show memory before the call
        print("Memory before call:")
        show_memory(mem)

        # 5) Build and display the payload
        payload = [
            {"role": role, "content": content}
            for role, content, _ in mem.query("", top_k=10)
        ] + [{"role": "user", "content": text}]
        print("Payload to OpenAI:")
        for msg in payload:
            print("  ", msg)

        # 6) Make the API call
        reply = adapter.call([{"role": "user", "content": text}])
        print("<<< Assistant:", reply)

        # 7) Show memory after the call
        print("Memory after call:")
        show_memory(mem)

    print("\n=== Conversation Complete ===")

if __name__ == "__main__":
    main()

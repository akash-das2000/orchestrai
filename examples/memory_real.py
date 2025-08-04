# examples/memory_real.py

import os
from dotenv import load_dotenv

# 1) Load your .env (must live at project root)
load_dotenv()

import openai
from google import genai
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore
from orchestrai.memory.adapters.openai_adapter import OpenAIAdapter

# 2) Configure each client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

openai.api_key = OPENAI_API_KEY
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# 3) Build a single memory store
mem = RollingBufferStore(max_size=10)

# 4) OpenAI adapter
openai_adapter = OpenAIAdapter(
    memory=mem,
    model="gpt-4o-mini",
    default_chat_kwargs={"temperature": 0.3, "max_tokens": 150}
)

# 5) (Later) you could write a GeminiAdapter exactly the same way
#    from orchestrai.memory.adapters.gemini_adapter import GeminiAdapter
#    gemini_adapter = GeminiAdapter(memory=mem, client=gemini_client, ...)

def demo_openai():
    print("=== OpenAI Demo ===")
    user_msg = {"role": "user", "content": "Hello, who are you?"}
    reply = openai_adapter.call([user_msg])
    print("Assistant (OpenAI):", reply)

def show_memory():
    print("\n=== Memory Buffer ===")
    for role, content, _ in mem.query("", top_k=10):
        print(f"  [{role}] {content}")

if __name__ == "__main__":
    demo_openai()
    show_memory()

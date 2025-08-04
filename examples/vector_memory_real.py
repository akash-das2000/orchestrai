# examples/vector_memory_conversation.py

import os
from dotenv import load_dotenv
import openai
from orchestrai.memory.stores.vector_memory import VectorMemoryStore

def main():
    # ——— 1) Load API key ———
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    # ——— 2) Create a vector‐based memory ———
    store = VectorMemoryStore(
        embed_model="text-embedding-ada-002",
        dim=1536
    )

    # ——— 3) Define your question‐asking routine ———
    def ask(question: str, idx: int) -> str:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": question}
            ],
            temperature=0.5,
            max_tokens=100
        ).choices[0].message.content

        # Store both Q & A
        store.add(f"q{idx}", question,   {"role": "user", "text": question})
        store.add(f"a{idx}", resp,       {"role": "assistant", "text": resp})
        return resp

    # ——— 4) Run a longer conversation ———
    questions = [
        "Where is the Great Pyramid of Giza located?",
        "Who wrote the novel ’1984’?",
        "What programming language is named after a comedy troupe?",
        "How many planets are there in our solar system?",
        "What’s the boiling point of water at sea level in Celsius?"
    ]

    for i, q in enumerate(questions, start=1):
        print(f"\nUser:      {q}")
        answer = ask(q, i)
        print(f"Assistant: {answer}")

    # ——— 5) Now do a semantic lookup ———
    lookup = "tell me about water boiling"
    print(f"\nSemantic query: “{lookup}”")
    hits = store.query(lookup, top_k=3)
    print("Top matches from memory:")
    for rank, (key, meta) in enumerate(hits, start=1):
        print(f"  {rank}. [{meta['role']}] {meta['text']}")

if __name__ == "__main__":
    main()

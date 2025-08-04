# examples/codebase_expert_demo.py

import os
import subprocess
from dotenv import load_dotenv
import openai
import re

from orchestrai.memory.stores.vector_memory import VectorMemoryStore
from orchestrai.memory.stores.summarizing_memory import SummarizingMemoryStore
from orchestrai.memory.stores.key_value_store import KeyValueStore
from orchestrai.memory.stores.rolling_buffer import RollingBufferStore

def load_code_docstrings(root_dir: str):
    """
    Walk the directory and extract (function_name, docstring) pairs
    from every .py file.
    """
    snippets = []
    pattern = re.compile(r'def\s+(\w+)\(.*?\):\s+"""(.*?)"""', re.DOTALL)
    for dirpath, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            text = open(path, "r", encoding="utf-8").read()
            for fn, doc in pattern.findall(text):
                key = f"{filename}:{fn}"
                snippets.append((key, doc.strip()))
    return snippets

def load_commit_logs(n: int = 30):
    """
    Grab the last n commit messages via git.
    """
    res = subprocess.run(
        ["git", "log", "--oneline", "-n", str(n)],
        capture_output=True, text=True, check=False
    )
    return res.stdout.splitlines()

def main():
    # ——— 1) Load API key ———
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    # ——— 2) Initialize memory stores ———
    # Semantic memory for code docstrings
    vector_store = VectorMemoryStore(embed_model="text-embedding-ada-002", dim=1536)
    # Commit‐log buffer + summarizer
    commit_buffer = RollingBufferStore(max_size=100)
    commit_memory = SummarizingMemoryStore(
        inner=commit_buffer,
        threshold=20,
        chunk_size=5,
        model="gpt-4o-mini",
        summarizer_kwargs={"temperature":0.3,"max_tokens":50}
    )
    # Key/value for style guides
    kv = KeyValueStore(db_path=":memory:")
    kv.set("style.max_line_length", "88")
    # Rolling buffer for recent interactive queries
    query_buffer = RollingBufferStore(max_size=10)

    # ——— 3) Index code docstrings ———
    snippets = load_code_docstrings(".")
    for key, doc in snippets:
        vector_store.add(key, doc, metadata={"type": "docstring"})
    print(f"Indexed {len(snippets)} function docstrings into semantic memory.")

    # ——— 4) Index recent commit messages ———
    commits = load_commit_logs(30)
    for c in commits:
        commit_memory.add("commit", c)
    print(f"Indexed {len(commits)} commits; running summary:")
    for _, content, _ in commit_buffer.query("", top_k=5):
        print("  •", content)

    # ——— 5) Simulate interactive queries ———
    questions = [
        "How do I initialize the database?",
        "What style rules should I follow?",
        "Show me the docstring for the API client function.",
    ]

    for q in questions:
        print(f"\nUser query: {q}")
        # 5a) Save to rolling buffer
        query_buffer.add("user", q)
        # 5b) Handle special KV lookup
        if "style" in q.lower():
            val = kv.get("style.max_line_length")
            resp = f"Max line length is set to {val} characters."
        else:
            # 5c) Semantic lookup in code docstrings
            hits = vector_store.query(q, top_k=3)
            resp = "Top matches:\n" + "\n".join(
                f"  - [{meta['type']}] {meta['type']} for {key}: {meta.get('text')}"
                for key, meta in hits
            )
        print("Assistant:", resp)
        query_buffer.add("assistant", resp)

    # ——— 6) Final: Semantic query on commit log ———
    followup = "What major refactors happened recently?"
    print(f"\nSemantic commit query: {followup}")
    # We embed and search commit_buffer entries directly
    hits = commit_buffer.query(followup, top_k=3)
    print("Recent commits / summaries:")
    for _, content, _ in hits:
        print("  •", content)

if __name__ == "__main__":
    main()

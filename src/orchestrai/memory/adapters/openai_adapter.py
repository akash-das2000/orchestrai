import openai
from typing import List, Dict, Any
from orchestrai.memory.core import MemoryStore

class OpenAIAdapter():
    """
    Wraps an OpenAI ChatCompletion call, 
    injecting your MemoryStore before sending and
    saving the assistant’s reply back into memory.
    
    Parameters:
      - memory: your MemoryStore implementation
      - model: the model name (e.g. "gpt-4o")
      - default_chat_kwargs: a dict of default parameters for Chat API
    """

    def __init__(
            self, 
            memory : MemoryStore,
            model: str = "gpt-4o-mini",
            default_chat_kwargs: Dict[str, Any] = None,
    ):
        self.memory = memory
        self.model = model
        self.default_chat_kwargs = default_chat_kwargs or {
            "temperature": 0.7,
            #add other defaults for max_tokens, top_p, etc.
        }

    def call(
            self, 
            messages: List[Dict[str,str]],
            chat_kwargs: Dict[str, Any] = None,
    ) -> str:
        """
        Send messages to OpenAI, with memory pre- and post-processing.

        Arguments:
        - messages: list of {"role": "...", "content": "..."} dicts
        - chat_kwargs: overrides or additions to default_chat_kwargs

        Returns:
        - assistant reply content
        """
        # 1) Persist incoming user/system messages into memory
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "system"):
                self.memory.add(role, content)

            
        # 2) Retrieve recent memory entries
        mem_entries = self.memory.query("", top_k = 10)
        mem_messages = [
            {"role": role, "content":content}
            for role, content, _ in mem_entries
        ]

        # 3) Combine memory + new messages
        payload = mem_messages + messages

        # 4) prepare chat parameters
        params = {**self.default_chat_kwargs}
        if chat_kwargs:
            params.update(chat_kwargs)

        # 5) Call openai endpoint
        resp = openai.chat.completions.create(
            model = self.model,
            messages = payload,
            **params,
        )
        reply = resp.choices[0].message.content

        # 6) Save the assistent's reply to memory
        self.memory.add("assistant", reply)


        return reply
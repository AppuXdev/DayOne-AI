"""Lightweight chat history helper used instead of deprecated LangChain memory classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ChatMessage:
    role: str
    content: str


class ConversationHistory:
    """Minimal in-process conversation store compatible with current usage.

    The app only needs:
    - `.chat_memory.messages`
    - `.chat_memory.add_user_message()`
    - `.chat_memory.add_ai_message()`
    - `.chat_memory.clear()`

    This keeps behavior stable while avoiding LangChain's deprecated memory API.
    """

    def __init__(self) -> None:
        self.messages: List[ChatMessage] = []

    @property
    def chat_memory(self) -> "ConversationHistory":
        return self

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    def clear(self) -> None:
        self.messages.clear()

# Copyright (c) 2024 Microsoft Corporation.


"""Ollama LLM implementations."""

from .config import OllamaConfig, PublicOllamaConfig
from .factories import (
    create_ollama_chat_llm,
    create_ollama_client,
    create_ollama_embeddings_llm,
)
from .roles import OllamaChatRole
from .types.client import (
    OllamaClient,
    OllamaEmbeddingsLLM,
    OllamaStreamingChatLLM,
    OllamaTextChatLLM,
)

# TODO: include type aliases?
__all__ = [
    "OllamaChatRole",
    "OllamaClient",
    "OllamaConfig",
    "OllamaConfig",
    "OllamaEmbeddingsLLM",
    "OllamaStreamingChatLLM",
    "OllamaTextChatLLM",
    "PublicOllamaConfig",
    "create_ollama_chat_llm",
    "create_ollama_client",
    "create_ollama_embeddings_llm",
]

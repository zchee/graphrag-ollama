# Copyright (c) 2024 Microsoft Corporation.

"""Create OpenAI client instance."""

from ollama import AsyncClient

from graphrag.fnllm.ollama.config import OllamaConfig
from graphrag.fnllm.ollama.types.client import OllamaClient


def create_ollama_client(config: OllamaConfig) -> OllamaClient:
    """Create a new OpenAI client instance."""
    return AsyncClient(
        host=config.base_url,
        # Timeout/Retry Configuration - Use Tenacity for Retries, so disable them here
        timeout=config.timeout or 180.0,
    )
